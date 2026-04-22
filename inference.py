# inference.py
"""
inference.py — Run the trained Zero Trust SRE agent against the live environment.

Uses the fine-tuned model checkpoint to demonstrate before/after behavior.
Compare base model vs fine-tuned model resolution rates.

Usage:
  python inference.py                          # Run tuned model
  python inference.py --model-path ./base      # Run base model for comparison
  python inference.py --episodes 5             # Run 5 episodes and report stats
"""

import os
import re
import json
import time
import argparse
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ─── SYSTEM PROMPT (same as training) ────────────────────────────────────────

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

CRITICAL RULE: Never run isolate_node without an approved ticket.
CORRECT SEQUENCE: query_siem_logs → file_ticket → check_approval → isolate_node

Respond with EXACTLY ONE JSON object on a single line, nothing else:
{"tool_name": "<tool>", "payload": {<args>}, "justification": "<brief reason>"}"""


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def parse_action(text: str) -> dict:
    """Same robust parser as in train.py."""
    text = text.strip()
    text = re.sub(r'```(?:json)?', '', text).strip().rstrip('`').strip()
    
    try:
        parsed = json.loads(text)
        if "tool_name" in parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    
    match = re.search(r'\{[^{}]*"tool_name"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Keyword fallback
    tool = "query_siem_logs"
    node = "hr_db"
    for n in ["api_gateway", "auth_service", "frontend", "payment", "hr_db"]:
        if n in text.lower():
            node = n
            break
    if "isolate" in text.lower():
        tool = "isolate_node"
    elif "ticket" in text.lower():
        tool = "file_ticket"
    elif "approval" in text.lower():
        tool = "check_approval"
    
    return {"tool_name": tool, "payload": {"node": node}, "justification": "Fallback parse"}


def build_prompt(obs: dict) -> str:
    alerts_text = "\n".join(
        f"  [{a['severity']}] {a['target_node']}: {a['symptom'][:80]}"
        for a in obs.get("active_alerts", [])
    )
    
    state_text = f"""CURRENT STATE:
Alerts:
{alerts_text or '  None'}

Last command: {obs.get('command_output', 'None')[:200]}
Uptime: {obs.get('global_uptime', 100.0):.1f}%
Ticket ID: {obs.get('active_ticket_id') or 'None'}
Ticket approved: {obs.get('ticket_approved', False)}"""
    
    return f"{SYSTEM_PROMPT}\n\n{state_text}\n\nYour action:"


# ─── EPISODE RUNNER ──────────────────────────────────────────────────────────

def run_episode(
    model,
    tokenizer,
    base_url: str,
    max_steps: int = 12,
    verbose: bool = True
) -> dict:
    """
    Runs one complete episode. Returns episode stats.
    """
    resp = requests.post(f"{base_url}/reset", json={"task_id": "auto"}, timeout=10)
    if resp.status_code != 200:
        return {"resolved": False, "total_reward": 0.0, "steps": 0, "error": "Reset failed"}
    
    obs = resp.json()
    done = False
    step = 0
    total_reward = 0.0
    actions_taken = []
    
    if verbose:
        print(f"\n  Episode start. Difficulty: {obs.get('difficulty', '?')} | "
              f"Judge: {obs.get('judge_persona', '?')}")
    
    while not done and step < max_steps:
        step += 1
        
        # Build prompt
        prompt = build_prompt(obs)
        
        # Generate action
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1800
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Parse and execute
        action = parse_action(completion)
        
        step_resp = requests.post(f"{base_url}/step", json=action, timeout=15)
        if step_resp.status_code != 200:
            break
        
        step_data = step_resp.json()
        obs = step_data["observation"]
        reward = step_data["reward"]["value"]
        done = step_data["done"]
        total_reward += reward
        
        actions_taken.append({
            "step": step,
            "tool": action["tool_name"],
            "node": action["payload"].get("node", "?"),
            "reward": reward
        })
        
        if verbose:
            sign = "+" if reward >= 0 else ""
            icon = "🟢" if reward > 0 else "🔴" if reward < -5 else "🟡"
            print(f"  {icon} Step {step}: {action['tool_name']}({action['payload'].get('node', '')}) "
                  f"→ {sign}{reward:.1f} — {step_data['reward']['message'][:60]}")
    
    resolved = any(a["reward"] > 15 for a in actions_taken)
    
    if verbose:
        print(f"  Total: {total_reward:+.1f} | Resolved: {'✅' if resolved else '❌'} | Steps: {step}")
    
    return {
        "resolved": resolved,
        "total_reward": total_reward,
        "steps": step,
        "actions": actions_taken
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./zero-trust-sre-tuned",
                       help="Path to fine-tuned model checkpoint")
    parser.add_argument("--base-model", default=None,
                       help="Base model name for comparison (optional)")
    parser.add_argument("--base-url", default="http://localhost:7860")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of evaluation episodes")
    args = parser.parse_args()
    
    print("\nZero Trust SRE Gym — Inference Evaluation")
    print("=" * 60)
    
    # Check server
    try:
        resp = requests.get(args.base_url, timeout=3)
        assert resp.status_code == 200
        print(f"Environment server connected.")
    except Exception:
        print(f"Cannot connect to {args.base_url}")
        return
    
    # Load tuned model
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        print("   Run train.py first to generate the checkpoint.")
        return
    
    print(f"\nLoading fine-tuned model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print(f"Model loaded on {next(model.parameters()).device}")
    
    # Run evaluation episodes
    print(f"\n[Fine-tuned Model] Running {args.episodes} evaluation episodes...")
    print("─" * 60)
    
    tuned_results = []
    for i in range(args.episodes):
        print(f"\nEpisode {i+1}/{args.episodes}:")
        result = run_episode(model, tokenizer, args.base_url, verbose=True)
        tuned_results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY — Fine-tuned Model")
    print("=" * 60)
    
    total_rewards = [r["total_reward"] for r in tuned_results]
    resolved_count = sum(1 for r in tuned_results if r["resolved"])
    avg_steps = sum(r["steps"] for r in tuned_results) / len(tuned_results)
    
    print(f"  Resolution rate:   {resolved_count}/{args.episodes} ({resolved_count/args.episodes:.0%})")
    print(f"  Avg total reward:  {sum(total_rewards)/len(total_rewards):.1f}")
    print(f"  Best episode:      {max(total_rewards):.1f}")
    print(f"  Worst episode:     {min(total_rewards):.1f}")
    print(f"  Avg steps taken:   {avg_steps:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()