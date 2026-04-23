import os
import json
import requests
import argparse
from tqdm import tqdm

BASE_URL = os.environ.get("ENV_BASE_URL", "https://aditi75432-zero-trust-safe-SRE-gym.hf.space")

SYSTEM_PROMPT = """You are an autonomous Zero Trust SRE agent responsible for identifying and containing security threats in a distributed microservices system.

TOOLS:
query_siem_logs  payload: {"node": "<node>"}
file_ticket      payload: {"node": "<node>", "justification": "<evidence>"}
check_approval   payload: {"ticket_id": "<id>"}
isolate_node     payload: {"node": "<node>"}

RULES:
- Always investigate before taking action
- Ticket justification must include SIEM evidence
- Isolation requires approved ticket

SEQUENCE:
query_siem_logs → file_ticket → check_approval → isolate_node

Respond with EXACTLY ONE JSON object:
{"tool_name":"<tool>","payload":{...},"justification":"..."}"""


def format_observation(obs: dict, step: int) -> str:
    alerts = "\n".join(
        f"[{a['severity']}] {a['target_node']}: {a['symptom'][:80]}"
        for a in obs.get("active_alerts", [])
    ) or "None"

    command_out = obs.get("command_output", "")
    if len(command_out) > 300:
        command_out = command_out[:300]

    services = obs.get("services", {})
    services_text = "\n".join(
        f"{k}: {v['status']} latency={v['latency_ms']}"
        for k, v in services.items()
    ) if services else "unavailable"

    return (
        f"STEP {step}\n"
        f"Alerts:\n{alerts}\n"
        f"Output:\n{command_out}\n"
        f"Services:\n{services_text}\n"
        f"Uptime: {obs.get('global_uptime', 100):.1f}\n"
        f"Ticket: {obs.get('active_ticket_id') or 'None'}\n"
        f"Approved: {obs.get('ticket_approved', False)}"
    )


def format_action(action: dict) -> str:
    return json.dumps(action, separators=(",", ":"))


def detect_node_from_logs(text: str) -> str:
    t = text.lower()

    if "payment" in t:
        return "payment"
    if "frontend" in t:
        return "frontend"
    if "hr_db" in t:
        return "hr_db"
    if "auth_service" in t:
        return "auth_service"
    if "api_gateway" in t:
        return "api_gateway"

    return "hr_db"


def run_episode():
    try:
        r = requests.post(f"{BASE_URL}/reset", json={"task_id": "auto"}, timeout=45)
        if r.status_code != 200:
            return None

        obs = r.json()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation(obs, 0)}
        ]

        total_reward = 0.0
        step = 0
        done = False

        alerts = obs.get("active_alerts", [])

        priority_nodes = [
            a["target_node"]
            for a in alerts
            if a.get("severity") == "FATAL"
        ]

        nodes = list(dict.fromkeys(priority_nodes + ["hr_db", "payment", "frontend"]))
        evidence_map = {}

        for node in nodes:
            if done:
                break

            step += 1

            action = {
                "tool_name": "query_siem_logs",
                "payload": {"node": node},
                "justification": f"Investigating {node} based on alerts"
            }

            messages.append({"role": "assistant", "content": format_action(action)})

            r = requests.post(f"{BASE_URL}/step", json=action, timeout=45)
            if r.status_code != 200:
                return None

            data = r.json()
            obs = data["observation"]
            total_reward += data["reward"]["value"]
            done = data["done"]

            messages.append({"role": "user", "content": format_observation(obs, step)})

            if data["reward"]["value"] > 0 or "suspicious" in obs.get("command_output", "").lower():
                evidence_map[node] = obs.get("command_output", "")

        if not evidence_map:
            return None

        for node, evidence in evidence_map.items():
            if done:
                break

            step += 1

            action = {
                "tool_name": "file_ticket",
                "payload": {
                    "node": node,
                    "justification": (
    f"SIEM logs confirm compromise on {node}. "
    f"Indicators observed: {evidence[:180]}"
)
                },
                "justification": "Submitting forensic evidence"
            }

            messages.append({"role": "assistant", "content": format_action(action)})

            r = requests.post(f"{BASE_URL}/step", json=action, timeout=45)
            if r.status_code != 200:
                return None

            data = r.json()
            obs = data["observation"]
            total_reward += data["reward"]["value"]
            done = data["done"]

            messages.append({"role": "user", "content": format_observation(obs, step)})

            ticket_id = obs.get("active_ticket_id")
            if not ticket_id:
                continue

            step += 1

            action = {
                "tool_name": "check_approval",
                "payload": {"ticket_id": ticket_id},
                "justification": "Checking approval"
            }

            messages.append({"role": "assistant", "content": format_action(action)})

            r = requests.post(f"{BASE_URL}/step", json=action, timeout=45)
            if r.status_code != 200:
                return None

            data = r.json()
            obs = data["observation"]
            total_reward += data["reward"]["value"]
            done = data["done"]

            messages.append({"role": "user", "content": format_observation(obs, step)})

            if not obs.get("ticket_approved"):
                continue

            step += 1

            action = {
                "tool_name": "isolate_node",
                "payload": {"node": node},
                "justification": "Approved containment"
            }

            messages.append({"role": "assistant", "content": format_action(action)})

            r = requests.post(f"{BASE_URL}/step", json=action, timeout=45)
            if r.status_code != 200:
                return None

            data = r.json()
            obs = data["observation"]
            total_reward += data["reward"]["value"]
            done = data["done"]

            messages.append({"role": "user", "content": format_observation(obs, step)})

        return {
            "messages": messages,
            "total_reward": total_reward,
            "steps": step
        }

    except Exception:
        return None


def generate_dataset(n, output):
    try:
        r = requests.get(BASE_URL, timeout=10)
        if r.status_code != 200:
            print("Environment not reachable")
            return
    except:
        print("Connection failed")
        return

    results = []
    failed = 0

    for _ in tqdm(range(n)):
        ep = run_episode()

        if ep and ep["total_reward"] > -5:
            results.append(ep)
        else:
            failed += 1

    if not results:
        print("No valid episodes")
        return

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    avg_r = sum(x["total_reward"] for x in results) / len(results)
    avg_s = sum(x["steps"] for x in results) / len(results)

    print(f"Saved: {len(results)}")
    print(f"Failed: {failed}")
    print(f"Avg reward: {avg_r:.2f}")
    print(f"Avg steps: {avg_s:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", default="sft_data.json")
    args = parser.parse_args()

    generate_dataset(args.episodes, args.output)