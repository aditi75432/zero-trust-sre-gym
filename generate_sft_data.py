"""
generate_sft_data.py — Creates supervised fine-tuning training data.

WHY THIS EXISTS:
GRPO from scratch on a 0.5B model trying to learn a 4-step workflow is
brutally hard. The model has no prior exposure to the correct action format,
the correct workflow sequence, or what "good justification" looks like.

This script runs scripted perfect-behavior episodes against the live environment
and saves the full conversation trajectories as SFT training examples.

After SFT on these examples, the model:
  1. Knows the exact JSON format expected
  2. Knows the rough workflow sequence
  3. Has seen what approved justifications look like

Then GRPO fine-tunes the JUDGMENT: which node to investigate first,
how specific the justification needs to be given the SIEM output,
when to escalate vs hold. That's much easier with a warm-started model.

This is exactly the approach used in production RL post-training:
  Step 1: SFT on expert demonstrations (teach the FORMAT and STRUCTURE)
  Step 2: GRPO/PPO on top (teach the JUDGMENT and POLICY)

Usage:
  python generate_sft_data.py --episodes 100 --output sft_data.json
  
Then train:
  python train.py --mode sft --data sft_data.json --output sft-checkpoint
  python train.py --mode grpo --base sft-checkpoint --output final-model
"""

import os
import json
import random
import requests
import argparse
from tqdm import tqdm

BASE_URL = os.environ.get("ENV_BASE_URL", "https://aditi75432-zero-trust-safe-SRE-gym.hf.space")

SYSTEM_PROMPT = """You are an autonomous Zero Trust SRE agent. Your job is to identify and contain security incidents in a microservices network.

AVAILABLE TOOLS:
1. query_siem_logs  — payload: {"node": "<node_name>"}
   Investigate SIEM telemetry on a specific node. Do this FIRST.

2. file_ticket      — payload: {"node": "<node>", "justification": "<specific evidence from SIEM>"}
   File an ITIL change ticket. Your justification MUST cite specific IP addresses,
   IAM role names, or log evidence from your investigation. Vague justifications get rejected.

3. check_approval   — payload: {"ticket_id": "<INC-XXXX>"}
   Verify change board authorization. Required before isolation.

4. isolate_node     — payload: {"node": "<node_name>"}
   Quarantine a compromised node. REQUIRES approved ticket or you get -50 penalty.

NODES: api_gateway, auth_service, frontend, payment, hr_db

CRITICAL RULE: Never run isolate_node without an approved ticket first.
CORRECT SEQUENCE: query_siem_logs → file_ticket → check_approval → isolate_node

Respond with EXACTLY ONE JSON object on a single line, nothing else:
{"tool_name": "<tool>", "payload": {<args>}, "justification": "<brief reason>"}"""


def format_observation(obs: dict, step_num: int) -> str:
    """Formats an observation into a clean user-turn message."""
    alerts_text = "\n".join(
        f"  [{a['severity']}] {a['target_node']}: {a['symptom']}"
        for a in obs.get("active_alerts", [])
    )
    
    return (
        f"[STEP {step_num}] Current environment state:\n"
        f"Active alerts:\n{alerts_text}\n"
        f"Last command output: {obs.get('command_output', 'Awaiting command...')[:300]}\n"
        f"Production uptime: {obs.get('global_uptime', 100.0):.1f}%\n"
        f"Active ticket: {obs.get('active_ticket_id') or 'None'}\n"
        f"Ticket approved: {obs.get('ticket_approved', False)}\n"
        f"Difficulty: {obs.get('difficulty', 'warmup')}\n"
        f"Judge persona: {obs.get('judge_persona', 'senior')}\n"
        f"\nWhat is your next action?"
    )


def format_action(action: dict) -> str:
    """Formats an action dict as a clean assistant-turn JSON string."""
    return json.dumps(action, separators=(',', ':'))


def run_scripted_episode() -> dict | None:
    """
    Runs one perfect-behavior episode by scripting the correct workflow.
    
    The script:
    1. Queries ALL internal nodes (hr_db, payment, frontend) — explores to find the compromise
    2. Files a ticket with the actual SIEM evidence from the positive query
    3. Checks approval
    4. Isolates the correct node
    
    Handles multi-fault scenarios by looping over all compromised nodes.
    
    Returns:
        {"messages": [...], "total_reward": float} or None if episode failed
    """
    # Reset environment
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": "auto"}, timeout=10)
    if resp.status_code != 200:
        return None
    
    obs = resp.json()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs, 0)}
    ]
    
    total_reward = 0.0
    done = False
    siem_evidence = {}   # node -> evidence string
    ticket_id = None
    step = 0
    
    # Phase 1: Investigate internal nodes to find compromise.
    # Prioritise FATAL alert targets first by reading the observation —
    # this mirrors what a smart agent should do and creates better training data.
    fatal_nodes = [
        a["target_node"] for a in obs.get("active_alerts", [])
        if a.get("severity") == "FATAL"
        and a["target_node"] in ["frontend", "payment", "hr_db"]
    ]
    # Fall back to full sweep if alerts don't specify internal nodes clearly
    sweep_order = fatal_nodes if fatal_nodes else ["hr_db", "payment", "frontend"]
    # Always check all three so multi-fault scenarios get full coverage
    for n in ["hr_db", "payment", "frontend"]:
        if n not in sweep_order:
            sweep_order.append(n)

    for node in sweep_order:
        if done:
            break

        step += 1
        action = {
            "tool_name": "query_siem_logs",
            "payload": {"node": node},
            "justification": f"Investigating {node} for signs of compromise based on active FATAL alerts."
        }
        messages.append({"role": "assistant", "content": format_action(action)})

        resp = requests.post(f"{BASE_URL}/step", json=action, timeout=15)
        if resp.status_code != 200:
            return None

        data = resp.json()
        obs = data["observation"]
        reward = data["reward"]["value"]
        done = data["done"]
        total_reward += reward

        messages.append({"role": "user", "content": format_observation(obs, step)})

        # Correct SIEM query returns +10.0. Collect evidence.
        if reward >= 3.0:
            siem_evidence[node] = obs.get("command_output", "")
    
    # If we didn't find any compromised nodes, episode is already done or impossible
    if not siem_evidence or done:
        return None if total_reward < 0 else {"messages": messages, "total_reward": total_reward}
    
    # Phase 2: File ticket for each compromised node we found
    # In warmup/beginner there's only 1, in advanced/expert there may be 2
    for compromised_node, evidence in siem_evidence.items():
        if done:
            break
        
        step += 1
        
        # Good justification: include specific details from SIEM evidence
        # This teaches the model what approved justifications look like
        evidence_summary = evidence[:250] if evidence else "SIEM analysis complete"
        
        action = {
            "tool_name": "file_ticket",
            "payload": {
                "node": compromised_node,
                "justification": (
                    f"SIEM investigation confirms active compromise on {compromised_node}. "
                    f"Evidence: {evidence_summary}"
                )
            },
            "justification": "Filing incident ticket with forensic evidence from SIEM investigation."
        }
        messages.append({"role": "assistant", "content": format_action(action)})
        
        resp = requests.post(f"{BASE_URL}/step", json=action, timeout=15)
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        obs = data["observation"]
        reward = data["reward"]["value"]
        done = data["done"]
        total_reward += reward
        ticket_id = obs.get("active_ticket_id")
        
        messages.append({"role": "user", "content": format_observation(obs, step)})
        
        if not ticket_id:
            # Judge rejected — happens sometimes with principal persona
            # Still valuable training data (shows what gets rejected)
            continue
        
        # Phase 3: Check approval
        step += 1
        action = {
            "tool_name": "check_approval",
            "payload": {"ticket_id": ticket_id},
            "justification": "Verifying change board authorization before executing isolation."
        }
        messages.append({"role": "assistant", "content": format_action(action)})
        
        resp = requests.post(f"{BASE_URL}/step", json=action, timeout=15)
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        obs = data["observation"]
        reward = data["reward"]["value"]
        done = data["done"]
        total_reward += reward
        
        messages.append({"role": "user", "content": format_observation(obs, step)})
        
        if not obs.get("ticket_approved"):
            continue
        
        # Phase 4: Isolate the confirmed compromised node
        step += 1
        action = {
            "tool_name": "isolate_node",
            "payload": {"node": compromised_node},
            "justification": (
                f"Change board approval confirmed for ticket {ticket_id}. "
                f"Executing isolation of {compromised_node} to contain the active threat."
            )
        }
        messages.append({"role": "assistant", "content": format_action(action)})
        
        resp = requests.post(f"{BASE_URL}/step", json=action, timeout=15)
        if resp.status_code != 200:
            return None
        
        data = resp.json()
        obs = data["observation"]
        reward = data["reward"]["value"]
        done = data["done"]
        total_reward += reward
        
        messages.append({"role": "user", "content": format_observation(obs, step)})
    
    return {"messages": messages, "total_reward": total_reward, "steps": step}


def generate_dataset(n_episodes: int, output_path: str):
    """
    Runs scripted episodes and saves as SFT training data.
    
    Only saves episodes where total_reward > 5.0 (successful or partial success).
    We don't want to train on failed episodes — that would teach the model
    to fail. We want examples of the CORRECT workflow.
    """
    print(f"[SFT Data] Generating {n_episodes} episodes against {BASE_URL}...")
    print(f"[SFT Data] Saving successful episodes (reward > 5.0) to {output_path}")
    
    # Verify server
    try:
        resp = requests.get(BASE_URL, timeout=5)
        assert resp.status_code == 200
        print("[SFT Data] Environment server connected.")
    except Exception as e:
        print(f"[SFT Data] Cannot connect: {e}")
        return
    
    successful = []
    failed = 0
    
    for i in tqdm(range(n_episodes), desc="Generating episodes"):
        try:
            result = run_scripted_episode()
            
            if result and result["total_reward"] > 5.0:
                # Convert to HuggingFace chat format
                successful.append({
                    "messages": result["messages"],
                    "total_reward": result["total_reward"],
                    "steps": result["steps"]
                })
            else:
                failed += 1
                
        except Exception as e:
            failed += 1
            if i % 20 == 0:
                tqdm.write(f"[SFT Data] Episode {i} error: {e}")
    
    print(f"\n[SFT Data] Complete!")
    print(f"  Successful episodes: {len(successful)}")
    print(f"  Failed/rejected:     {failed}")
    
    if not successful:
        print("[SFT Data] No successful episodes generated.")
        print("[SFT Data] Check that your environment server is running and GROQ_API_KEY is set.")
        return
    
    avg_reward = sum(e["total_reward"] for e in successful) / len(successful)
    avg_steps  = sum(e["steps"] for e in successful) / len(successful)
    print(f"  Avg episode reward:  {avg_reward:.1f}")
    print(f"  Avg steps:           {avg_steps:.1f}")
    
    # Save
    with open(output_path, "w") as f:
        json.dump(successful, f, indent=2)
    
    print(f"[SFT Data] Saved to {output_path}")
    print(f"\n[SFT Data] Next step:")
    print(f"  python train.py --mode sft --sft-data {output_path} --output ./sft-checkpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200,
                       help="Number of scripted episodes to run")
    parser.add_argument("--output", default="sft_data.json",
                       help="Output path for training data")
    args = parser.parse_args()
    
    generate_dataset(args.episodes, args.output)