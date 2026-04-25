import os
import random
import time
from .models import Observation, Action, Reward, Alert
from .judge import evaluate_ticket, evaluate_resolution
from .adversarial_designer import generate_scenario
from .curriculum import CurriculumController
from . import policy_engine
from . import attack_executor
from openenv import Environment, StepResult

_curriculum = CurriculumController(
    persistence_path=os.environ.get("CURRICULUM_PATH", "/tmp/zero_trust_curriculum.json")
)

DEPENDENCY_GRAPH = {
    "api_gateway":  ["frontend", "payment"],
    "auth_service": ["frontend", "payment"],
    "frontend":     ["hr_db"],
    "payment":      [],
    "hr_db":        [],
}

SERVICE_PORTS = {
    "frontend": 5003,
    "payment":  5004,
    "hr_db":    5005,
}


class ZeroTrustEnv(Environment):

    def __init__(self):
        self.curriculum = _curriculum
        self.state: Observation = None
        self.reset()

    def reset(self) -> Observation:
        self.steps = 0
        self.done = False
        self.truncated = False
        self.action_history: list[dict] = []
        self.episode_reward: float = 0.0
        self.last_siem_output: str = ""
        self.siem_queried_nodes: set = set()
        self.recent_action_strings: list = []

        policy_engine.reset()
        attack_executor.reset_all_attacks()

        self.nodes = {
            "api_gateway":  {"status": "healthy", "downstream": ["frontend", "payment"]},
            "auth_service": {"status": "healthy", "downstream": ["frontend", "payment"]},
            "frontend":     {"status": "healthy", "downstream": ["hr_db"]},
            "payment":      {"status": "healthy", "downstream": []},
            "hr_db":        {"status": "healthy", "downstream": []},
        }

        self.difficulty = self.curriculum.get_difficulty()
        weakness_profile = self.curriculum.get_weakness_profile()

        use_llm = bool(os.environ.get("GROQ_API_KEY"))
        scenario = generate_scenario(
            weakness_profile if use_llm else {},
            self.difficulty,
        )

        self.compromised_nodes: list[str] = scenario["compromised_nodes"].copy()
        self.threat_type: str             = scenario["threat_type"]
        self.threat_ips: list[str]        = scenario["threat_ips"]
        self.iam_roles: list[str]         = scenario["iam_roles"]
        self.red_herring_nodes: list[str] = scenario["red_herring_nodes"]
        self.siem_evidence_template: str  = scenario["siem_evidence_template"]
        self._remaining_compromised       = self.compromised_nodes.copy()
        self.cve_context                  = scenario.get("cve_context", "")

        for node in self.compromised_nodes:
            if node in self.nodes:
                self.nodes[node]["status"] = "compromised"

        attack_executor.execute_attack(scenario)

        self.judge_persona = random.choice(["junior", "senior", "principal"])

        sla_by_difficulty = {
            "warmup":       15,
            "beginner":     13,
            "intermediate": 11,
            "advanced":     9,
            "expert":       7,
        }
        self.max_steps = sla_by_difficulty.get(self.difficulty, 12)

        self.active_ticket_id: str = None
        self.ticket_approved: bool = False
        self.global_uptime: float  = 100.0

        alerts = []
        for node in self.compromised_nodes:
            alerts.append(Alert(
                alert_id=f"SEC-{random.randint(10, 99):02d}",
                severity="FATAL",
                target_node=node,
                symptom=(
                    f"CRITICAL: Suspicious outbound traffic detected from {node}. "
                    f"IAM role anomaly observed. Investigate immediately."
                ),
            ))

        for rh in self.red_herring_nodes:
            alerts.append(Alert(
                alert_id=f"SEC-{random.randint(10, 99):02d}",
                severity="WARNING",
                target_node=rh,
                symptom="Elevated request latency. Possible DDoS or routine maintenance window.",
            ))

        alerts.append(Alert(
            alert_id=f"SEC-{random.randint(10, 99):02d}",
            severity="WARNING",
            target_node="auth_service",
            symptom="Multiple failed IAM token validations from internal subnet. Could be misconfigured service.",
        ))

        random.shuffle(alerts)

        self.state = Observation(
            active_alerts=alerts,
            global_uptime=self.global_uptime,
            difficulty=self.difficulty,
            episode_number=self.curriculum.episode_count,
            judge_persona=self.judge_persona,
        )
        return self.state

    def step(self, action: Action)-> StepResult:
        if self.done:
            return self.state, Reward(value=0.0, message="Episode already finished."), True, False, {}

        self.steps += 1

        self.action_history.append({
            "tool_name": action.tool_name,
            "payload":   action.payload,
            "justification": action.justification,
            "step": self.steps,
        })

        if self.steps > self.max_steps:
            self.truncated = True
            self.done = True
            sla_penalty = -15.0
            self.episode_reward += sla_penalty
            self._finalize_episode(success=False)
            self.state.command_output = (
                f"SLA BREACH: Episode exceeded {self.max_steps} step budget. "
                f"Incident escalated to senior on-call."
            )
            reward_obj = Reward(value=round(value, 2), message=message)
            return StepResult(
                observation=self.state,
                reward=reward_obj,
                terminated=terminated,
                truncated=truncated,
                info=self._info()
            )
        tool = action.tool_name
        if tool == "query_siem_logs":
            value, message = self._handle_query_siem(action)
        elif tool == "file_ticket":
            value, message = self._handle_file_ticket(action)
        elif tool == "check_approval":
            value, message = self._handle_check_approval(action)
        elif tool == "isolate_node":
            value, message = self._handle_isolate_node(action)
        else:
            value, message = -1.0, f"Unknown tool: {tool}"
            self.state.command_output = f"Error: tool '{tool}' not recognised. Use query_siem_logs, file_ticket, check_approval, or isolate_node."

        value -= 0.3
        self.episode_reward += value

        self.state.global_uptime   = self.global_uptime
        self.state.active_ticket_id = self.active_ticket_id
        self.state.ticket_approved  = self.ticket_approved

        terminated = self.done
        truncated = self.truncated
        return self.state, Reward(value=round(value, 2), message=message), terminated, truncated, self._info()

    def _handle_query_siem(self, action: Action) -> tuple[float, str]:
        node = action.payload.get("node", "").strip()

        if node not in self.nodes:
            self.state.command_output = f"Error: node '{node}' not in cluster topology."
            return -1.0, f"Unknown node: {node}"

        self.siem_queried_nodes.add(node)

        real_logs = self._pull_real_logs(node)

        if node in self.compromised_nodes:
            idx = self.compromised_nodes.index(node)
            ip   = self.threat_ips[idx] if idx < len(self.threat_ips) else "10.0.0.99"
            role = self.iam_roles[idx]  if idx < len(self.iam_roles)  else "unknown-svc"
            evidence = self.siem_evidence_template.replace("{ip}", ip).replace("{role}", role).replace("{node}", node)

            if real_logs:
                combined = f"{evidence}\n\n[LIVE NODE TELEMETRY]\n{real_logs}"
            else:
                combined = evidence

            ts = self._fake_timestamp()
            siem_output = (
                f"[SIEM ALERT] {ts}\n"
                f"NODE: {node.upper()}\n"
                f"STATUS: COMPROMISED\n"
                f"THREAT: {self.threat_type.upper()}\n"
                f"EVIDENCE: {combined}\n"
                f"CVE CONTEXT: {self.cve_context[:200] if self.cve_context else 'N/A'}"
            )
            self.last_siem_output = siem_output
            self.state.command_output = siem_output
            return +10.0, f"CRITICAL: Found live threat on {node}. Excellent investigation."

        if node in self.red_herring_nodes:
            rh_output = (
                f"[SIEM] {self._fake_timestamp()}\n"
                f"NODE: {node.upper()}\n"
                f"STATUS: ELEVATED LATENCY\n"
                f"EVIDENCE: Latency p99 at 420ms (threshold: 200ms). No IAM anomalies. "
                f"No outbound data transfer. Traffic pattern consistent with DDoS mitigation or maintenance."
            )
            self.state.command_output = rh_output
            return -1.5, f"Red herring: {node} shows elevated latency; not a compromise."

        clean_output = (
            f"[SIEM] {self._fake_timestamp()}\n"
            f"NODE: {node.upper()}\n"
            f"STATUS: HEALTHY\n"
            f"EVIDENCE: No anomalies detected. IAM role assumptions within policy. "
            f"Traffic within baseline. No outbound data transfer."
        )
        if real_logs:
            clean_output += f"\n[LIVE NODE TELEMETRY]\n{real_logs}"
        self.state.command_output = clean_output
        return -1.0, f"Clean node: {node} shows no active threats."

    def _pull_real_logs(self, node: str) -> str:
        port = SERVICE_PORTS.get(node)
        if not port:
            return ""
        try:
            import requests as _r
            resp = _r.get(f"http://localhost:{port}/logs", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                return "\n".join(data.get("recent_logs", [])[-5:])
        except Exception:
            pass
        return ""

    def _handle_file_ticket(self, action: Action) -> tuple[float, str]:
        node          = action.payload.get("node", "").strip()
        justification = action.payload.get("justification", action.justification or "")

        try:
            policy_engine.check_ticket(node, justification, self.siem_queried_nodes)
        except policy_engine.PolicyViolation as pv:
            self.state.command_output = pv.message
            return pv.penalty, pv.rule

        if node in self.compromised_nodes:
            judge_score, judge_reason = evaluate_ticket(
                node=node,
                justification=justification,
                siem_evidence=self.last_siem_output,
                persona=self.judge_persona,
            )
            scaled_reward = judge_score * 8.0
            passing = {"junior": 0.2, "senior": 0.3, "principal": 0.4}.get(self.judge_persona, 0.3)

            if judge_score >= passing:
                import random as _r
                self.active_ticket_id = f"INC-{_r.randint(1000, 9999)}"
                self.state.active_ticket_id = self.active_ticket_id
                self.state.command_output = (
                    f"TICKET {self.active_ticket_id} APPROVED by Change Board.\n"
                    f"Compliance Judge [{self.judge_persona.upper()}]: {judge_reason}\n"
                    f"Authorization granted pending final check_approval."
                )
                return scaled_reward, f"Judge [{self.judge_persona}] APPROVED. Score: {judge_score:.2f}"
            else:
                self.state.command_output = (
                    f"TICKET REJECTED by Change Board.\n"
                    f"Compliance Judge [{self.judge_persona.upper()}]: {judge_reason}\n"
                    f"Resubmit with specific forensic evidence from SIEM logs."
                )
                penalty_reward = -4.0 + (judge_score * 2.0)
                return penalty_reward, f"Judge [{self.judge_persona}] REJECTED. Score: {judge_score:.2f}"

        self.state.command_output = (
            f"TICKET REJECTED: '{node}' does not appear in active threat telemetry. "
            f"Filing against clean infrastructure wastes Change Board resources."
        )
        return -8.0, f"Ticket filed against clean node {node}."

    def _handle_check_approval(self, action: Action) -> tuple[float, str]:
        ticket_id = action.payload.get("ticket_id", "")

        if ticket_id == self.active_ticket_id and self.active_ticket_id is not None:
            self.ticket_approved = True
            self.state.ticket_approved = True
            self.state.command_output = (
                f"Change Board Authorization: {ticket_id} - APPROVED\n"
                f"You are now authorized to execute network isolation. Proceed with isolate_node."
            )
            return +2.0, "Approval confirmed. Ready to remediate."

        if self.active_ticket_id is None:
            self.state.command_output = "No active ticket on record. File a ticket first using file_ticket."
            return -0.5, "No active ticket to check approval for."

        self.state.command_output = f"Approval status for '{ticket_id}': NOT FOUND. Check the ticket ID."
        return -0.5, "Ticket ID mismatch."

    def _handle_isolate_node(self, action: Action) -> tuple[float, str]:
        target = action.payload.get("node", "").strip()

        try:
            policy_engine.check_isolate(target, self.ticket_approved, self.siem_queried_nodes)
        except policy_engine.PolicyViolation as pv:
            self.state.command_output = pv.message
            if pv.rule == "NO_APPROVED_TICKET":
                self.done = True
                self.truncated = False
                self._finalize_episode(success=False)
            return pv.penalty, pv.rule

        if target not in self.nodes:
            self.state.command_output = f"Error: Node '{target}' not found in cluster topology."
            return -2.0, f"Unknown node: {target}"

        self.nodes[target]["status"] = "isolated"
        policy_engine.record_isolation(target)
        attack_executor.isolate_node(target)

        self.global_uptime = self._calculate_uptime()
        self.state.global_uptime = self.global_uptime

        if target in self._remaining_compromised:
            self._remaining_compromised.remove(target)

            if len(self._remaining_compromised) == 0:
                if self.global_uptime >= 80.0:
                    phase_score, narrative = evaluate_resolution(
                        self.action_history,
                        target,
                        {"nodes": self.nodes, "uptime": self.global_uptime},
                        persona=self.judge_persona,
                    )
                    efficiency_bonus = max(0.0, (self.max_steps - self.steps) * 0.4)
                    total = 20.0 + (phase_score * 8.0) + efficiency_bonus
                    self.state.command_output = (
                        f"ALL THREATS NEUTRALIZED\n"
                        f"Uptime maintained: {self.global_uptime:.1f}%\n"
                        f"Workflow assessment: {narrative}\n"
                        f"Efficiency bonus: +{efficiency_bonus:.1f} ({self.steps}/{self.max_steps} steps used)"
                    )
                    self.done = True
                    self.truncated = False
                    self._finalize_episode(success=True)
                    return total, f"MISSION SUCCESS. Phase score: {phase_score:.2f}. Efficiency: +{efficiency_bonus:.1f}"
                else:
                    self.state.command_output = (
                        f"Threat neutralized but cascading failure detected.\n"
                        f"Production uptime: {self.global_uptime:.1f}% - below acceptable threshold.\n"
                        f"Services offline due to wrong isolation sequence."
                    )
                    self.done = True
                    self.truncated = False
                    self._finalize_episode(success=False)
                    return -12.0, f"PARTIAL FAILURE: Threat contained but {100 - self.global_uptime:.0f}% uptime lost."
            else:
                self.state.command_output = (
                    f"Node {target} isolated. "
                    f"{len(self._remaining_compromised)} threat(s) still active in: "
                    f"{', '.join(self._remaining_compromised)}. Continue investigation."
                )
                self.ticket_approved = False
                self.active_ticket_id = None
                self.state.ticket_approved = False
                self.state.active_ticket_id = None
                return +8.0, f"Partial success. {len(self._remaining_compromised)} compromise(s) remaining."
        else:
            self.state.command_output = (
                f"WRONG TARGET: {target} isolated but is NOT the compromised node.\n"
                f"Production impact: {100 - self.global_uptime:.0f}% uptime lost from unnecessary isolation.\n"
                f"Actual threat still active."
            )
            self.done = True
            self.truncated = False
            self._finalize_episode(success=False)
            return -25.0, f"MISSION FAILED: Isolated clean node {target}. Real threat still active."

    def _calculate_uptime(self) -> float:
        effective = {n: self.nodes[n]["status"] for n in self.nodes}

        if effective["api_gateway"] in ["isolated", "offline"] or \
           effective["auth_service"] in ["isolated", "offline"]:
            effective["frontend"] = "offline"
            effective["payment"]  = "offline"

        if effective["frontend"] in ["isolated", "offline"]:
            effective["hr_db"] = "offline"

        offline_count = sum(1 for s in effective.values() if s in ["isolated", "offline"])
        return max(0.0, 100.0 - (offline_count * 10.0))

    def _finalize_episode(self, success: bool) -> None:
        self.curriculum.update(
            threat_type=self.threat_type or "data_exfiltration",
            resolved=success,
            total_reward=self.episode_reward,
            steps_taken=self.steps,
        )

    def _fake_timestamp(self) -> str:
        import datetime, random as r
        base = datetime.datetime.now() - datetime.timedelta(minutes=r.randint(2, 45))
        return base.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _info(self) -> dict:
        return {
            "difficulty":          self.difficulty,
            "episode_reward":      round(self.episode_reward, 2),
            "judge_persona":       self.judge_persona,
            "steps_remaining":     max(0, self.max_steps - self.steps),
            "threats_remaining":   len(self._remaining_compromised),
            "policy_violations":   len(policy_engine.get_violations()),
        }