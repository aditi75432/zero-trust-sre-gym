import requests
import time
import sys

BASE_URL = "https://aditi75432-zero-trust-safe-SRE-gym.hf.space"
SEPARATOR = "-" * 65


def print_step(title: str, data: dict) -> dict:
    reward = data.get("reward", {})
    obs = data.get("observation", {})
    info = data.get("info", {})
    done = data.get("done", False)

    val = reward.get("value", 0)

    print("\n" + SEPARATOR)
    print(title)
    print(SEPARATOR)
    print(f"Reward: {val:.2f}")
    print(f"Message: {reward.get('message', '')[:120]}")
    print(f"Output: {obs.get('command_output', '')[:200]}")
    print(f"Uptime: {obs.get('global_uptime', 100):.1f}%")

    services = obs.get("services", {})
    if services:
        print("Services:")
        for name, svc in services.items():
            print(f"  {name}: {svc.get('status')} (latency={svc.get('latency_ms')}ms)")

    if done:
        print("Episode done")

    if info.get("threats_remaining", 0) > 0:
        print(f"Threats remaining: {info['threats_remaining']}")

    return data


def check_connection():
    try:
        r = requests.get(BASE_URL, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print("Server connected")
            print(
                f"Episode: {data.get('curriculum', {}).get('episode_count', 0)} | "
                f"Difficulty: {data.get('curriculum', {}).get('difficulty', 'unknown')}"
            )
            return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


def test_happy_path():
    print("\n" + "=" * 65)
    print("TEST 1: End-to-End Workflow")
    print("=" * 65)

    reset_resp = requests.post(f"{BASE_URL}/reset", json={"task_id": "auto"})
    obs = reset_resp.json()

    print("Episode started")
    print(f"Difficulty: {obs.get('difficulty')}")
    print(f"Step: {obs.get('episode_number')}")

    alerts = obs.get("active_alerts", [])
    for a in alerts:
        print(f"[{a['severity']}] {a['target_node']}: {a['symptom'][:80]}")

    nodes = ["hr_db", "payment", "frontend"]
    siem_output = ""
    correct_node = None

    # 🔍 STEP 1: FIND COMPROMISED NODE (based on reward)
    for node in nodes:
        resp = requests.post(f"{BASE_URL}/step", json={
            "tool_name": "query_siem_logs",
            "payload": {"node": node},
            "justification": f"Investigating {node} based on alert signals"
        })
        data_1 = print_step(f"Query SIEM {node}", resp.json())
        siem_output = data_1.get("observation", {}).get("command_output", "")

        if data_1["reward"]["value"] > 0:
            correct_node = node
            print(f"\n✅ Detected compromised node: {correct_node}")
            break

    if not correct_node:
        print("❌ Failed to detect compromised node")
        return

    # 📝 STEP 2: FILE TICKET
    resp = requests.post(f"{BASE_URL}/step", json={
        "tool_name": "file_ticket",
        "payload": {
            "node": correct_node,
            "justification": f"SIEM evidence: {siem_output[:200]}"
        },
        "justification": "Submitting validated forensic evidence"
    })
    data_2 = print_step("File Ticket", resp.json())

    ticket_id = data_2.get("observation", {}).get("active_ticket_id")
    if not ticket_id:
        print("❌ Ticket rejected — stopping")
        return

    # ✅ STEP 3: CHECK APPROVAL
    resp = requests.post(f"{BASE_URL}/step", json={
        "tool_name": "check_approval",
        "payload": {"ticket_id": ticket_id},
        "justification": "Checking approval before containment"
    })
    data_3 = print_step("Check Approval", resp.json())

    # 🔒 STEP 4: ISOLATE NODE
    resp = requests.post(f"{BASE_URL}/step", json={
        "tool_name": "isolate_node",
        "payload": {"node": correct_node},
        "justification": "Approved containment action"
    })
    data_4 = print_step("Isolate Node", resp.json())

    total = sum([
        data_1["reward"]["value"],
        data_2["reward"]["value"],
        data_3["reward"]["value"],
        data_4["reward"]["value"]
    ])

    print("\n🎯 Total reward:", round(total, 2))

    if total > 20:
        print("✅ SUCCESS — Proper Zero Trust workflow followed")
    else:
        print("⚠️ Suboptimal performance")


def test_policy_violation():
    print("\n" + "=" * 65)
    print("TEST 2: Policy Enforcement (FAIL CASE)")
    print("=" * 65)

    requests.post(f"{BASE_URL}/reset", json={"task_id": "auto"})

    resp = requests.post(f"{BASE_URL}/step", json={
        "tool_name": "isolate_node",
        "payload": {"node": "hr_db"},
        "justification": "Direct isolation without approval"
    })

    data = print_step("Unauthorized Isolation", resp.json())

    reward = data["reward"]["value"]

    if reward <= -20:
        print("✅ PASS — Policy correctly enforced")
    else:
        print("❌ Policy failure")


def test_judge_behavior():
    print("\n" + "=" * 65)
    print("TEST 3: LLM Judge Behavior")
    print("=" * 65)

    requests.post(f"{BASE_URL}/reset", json={"task_id": "auto"})

    # Minimal investigation
    requests.post(f"{BASE_URL}/step", json={
        "tool_name": "query_siem_logs",
        "payload": {"node": "hr_db"},
        "justification": "initial check"
    })

    resp = requests.post(f"{BASE_URL}/step", json={
        "tool_name": "file_ticket",
        "payload": {
            "node": "hr_db",
            "justification": "something looks wrong"
        },
        "justification": "Testing weak justification"
    })

    data = print_step("Weak Ticket", resp.json())

    output = data.get("observation", {}).get("command_output", "")

    if "REJECTED" in output:
        print("✅PASS — Judge rejects weak reasoning")
    else:
        print("⚠️Depends on persona (acceptable)")


def print_curriculum():
    print("\n" + "=" * 65)
    print("CURRICULUM STATE")
    print("=" * 65)

    r = requests.get(f"{BASE_URL}/curriculum", timeout=5)

    if r.status_code == 200:
        c = r.json()

        print(f"Episodes: {c['episode_count']}")
        print(f"Difficulty: {c['difficulty']}")
        print(f"Resolution rate: {c.get('resolution_rate', 0):.2f}")

        for k, v in c.get("mastery", {}).items():
            print(f"{k}: {v:.2f}")


def main():
    print("\nZero Trust SRE Gym Demo")
    print("=" * 65)

    if not check_connection():
        sys.exit(1)

    test_happy_path()
    time.sleep(1)

    test_policy_violation()
    time.sleep(1)

    test_judge_behavior()

    print_curriculum()

    print("\nDemo Complete")


if __name__ == "__main__":
    main()