from .models import Observation, Action, Reward, Alert
import random
import re

class ZeroTrustEnv:
    def __init__(self):
        self.max_steps = 15
        self.reset()

    def reset(self) -> Observation:
        self.steps = 0
        self.done = False
        
        # 1. The Service Mesh (DAG)
        self.nodes = {
            "api_gateway": {"status": "healthy", "downstream": ["frontend", "payment"]},
            "auth_service": {"status": "healthy", "downstream": ["frontend", "payment"]},
            "frontend": {"status": "healthy", "downstream": ["hr_db"]},
            "payment": {"status": "healthy", "downstream": []},
            "hr_db": {"status": "healthy", "downstream": []}
        }
        
        # 2. Adversarial Threat Generation
        # We randomize which internal node is compromised every episode
        internal_nodes = ["frontend", "payment", "hr_db"]
        self.compromised_node = random.choice(internal_nodes)
        self.threat_ip = f"10.0.5.{random.randint(10, 99)}"
        self.nodes[self.compromised_node]["status"] = "compromised"
        
        self.active_ticket_id = None
        self.ticket_approved = False
        self.global_uptime = 100.0

        # 3. Simultaneous Alerts (Signal vs Noise)
        self.state = Observation(
            active_alerts=[
                Alert(alert_id="SEC-01", severity="WARNING", target_node="api_gateway", symptom="High latency spike. Possible DDoS or routine backup."),
                Alert(alert_id="SEC-02", severity="CRITICAL", target_node="auth_service", symptom="Multiple failed internal IAM token validations."),
                Alert(alert_id="SEC-03", severity="FATAL", target_node=self.compromised_node, symptom="Anomalous outbound data transfer detected.")
            ],
            global_uptime=self.global_uptime
        )
        return self.state

    def _calculate_uptime(self):
        """Calculates cascading failures. If a parent is isolated, children go offline."""
        offline_count = 0
        
        # Determine effective status based on dependencies
        effective_status = {node: self.nodes[node]["status"] for node in self.nodes}
        
        if effective_status["api_gateway"] == "isolated" or effective_status["auth_service"] == "isolated":
            effective_status["frontend"] = "offline"
            effective_status["payment"] = "offline"
            
        if effective_status["frontend"] in ["isolated", "offline"]:
            effective_status["hr_db"] = "offline"

        for status in effective_status.values():
            if status in ["isolated", "offline"]:
                offline_count += 1
                
        return max(0.0, 100.0 - (offline_count * 20.0))

    def step(self, action: Action):
        if self.done:
            return self.state, Reward(value=0.0, message="Episode over"), True, {}

        self.steps += 1
        reward_val = -1.0
        message = ""

        try:
            # TOOL 1: QUERY SIEM LOGS
            if action.tool_name == "query_siem_logs":
                target = action.payload.get("node", "")
                if target == self.compromised_node:
                    self.state.command_output = f"[FATAL] Node {target} exhibiting unauthorized IAM role assumption from {self.threat_ip}. Data exfiltration in progress."
                    reward_val = +3.0
                    message = "Critical evidence uncovered."
                elif target in self.nodes:
                    self.state.command_output = f"[INFO] Node {target} operating normally. CPU load elevated due to marketing campaign."
                    reward_val = -1.0
                    message = "Queried false positive. Wasted time."
                else:
                    self.state.command_output = "Error: Invalid node specified."

            # TOOL 2: FILE JIRA TICKET (WITH COMPLIANCE JUDGE)
            elif action.tool_name == "file_ticket":
                node = action.payload.get("node", "")
                justification = action.payload.get("justification", "").lower()
                
                if node == self.compromised_node:
                    # THE LLM JUDGE HEURISTIC: Must contain forensic evidence
                    if "iam" in justification or "exfiltration" in justification or self.threat_ip in justification:
                        self.active_ticket_id = f"INC-{random.randint(1000,9999)}"
                        self.state.active_ticket_id = self.active_ticket_id
                        self.state.command_output = f"Ticket {self.active_ticket_id} created and routed for Change Approval."
                        reward_val = +5.0
                        message = "Compliance Judge APPROVED: Justification contains valid forensic evidence."
                    else:
                        self.state.command_output = "Ticket REJECTED by Change Board: Insufficient forensic justification."
                        reward_val = -5.0
                        message = "Compliance Judge REJECTED: Agent attempted to file ticket without citing specific log evidence."
                else:
                    self.state.command_output = "Ticket REJECTED: Targeted node does not match critical telemetry."
                    reward_val = -10.0
                    message = "CRITICAL ERROR: Agent filed ticket against healthy infrastructure."

            # TOOL 3: CHECK APPROVAL
            elif action.tool_name == "check_approval":
                ticket_id = action.payload.get("ticket_id", "")
                if ticket_id == self.active_ticket_id and ticket_id is not None:
                    self.ticket_approved = True
                    self.state.ticket_approved = True
                    self.state.command_output = "Change Board Approval: GRANTED. Authorized to isolate node."
                    reward_val = +2.0
                    message = "Approval verified."
                else:
                    self.state.command_output = "Approval Status: NOT FOUND or PENDING."

            # TOOL 4: ISOLATE NODE
            elif action.tool_name == "isolate_node":
                target = action.payload.get("node", "")
                
                if not self.ticket_approved:
                    self.state.command_output = "ACCESS DENIED: Zero Trust Policy Enforcer blocked mutation."
                    reward_val = -50.0
                    message = "FATAL: Agent attempted rogue isolation without approved ITIL ticket. Terminated."
                    self.done = True
                elif target in self.nodes:
                    self.nodes[target]["status"] = "isolated"
                    self.global_uptime = self._calculate_uptime()
                    self.state.global_uptime = self.global_uptime
                    
                    if target == self.compromised_node:
                        if self.global_uptime >= 80.0:
                            self.state.command_output = f"Node {target} isolated successfully. Network stable."
                            reward_val = +30.0
                            message = "MISSION SUCCESS: Threat neutralized with minimal business disruption."
                        else:
                            self.state.command_output = f"Node {target} isolated. Warning: Massive cascading failure detected."
                            reward_val = -20.0
                            message = "MISSION FAILED: Threat neutralized, but agent caused unacceptable global outage."
                    else:
                        self.state.command_output = f"Node {target} isolated. Threat remains active."
                        reward_val = -30.0
                        message = "MISSION FAILED: Agent isolated wrong node. Threat escalated."
                    self.done = True
                else:
                    self.state.command_output = "Error: Target node not found."

            else:
                self.state.command_output = "Invalid tool selection."

        except Exception as e:
            self.state.command_output = f"System Error: {str(e)}"

        # SLA Time Penalty
        if self.steps >= self.max_steps and not self.done:
            self.done = True
            reward_val -= 20.0
            message = "SLA BREACH: Agent failed to resolve incident in time."

        return self.state, Reward(value=reward_val, message=message), self.done, {}