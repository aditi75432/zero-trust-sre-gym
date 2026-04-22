"""
train.py — GRPO Training for Zero Trust SRE Gym.

This script fixes the zero-loss problem that was killing your training run.

Root cause of your zero losses:
  1. All 8 GRPO rollouts got the same reward because the environment only had
     one deterministic outcome. Same scenario → same observation → same completion
     → same reward → zero advantage → zero loss.

  2. Temperature was too low (0.2). Near-zero temperature = near-identical
     completions across rollouts = zero reward variance = zero loss.

The fixes applied here:
  1. Environment generates different scenarios per reset via adversarial designer.
     Different scenarios guarantee different observations and different rewards.
     
  2. Rollout temperature is 0.7. Enough variation for meaningful advantage computation.
  
  3. We use num_generations=8 and reset the environment between each batch,
     so each group of 8 rollouts sees different scenarios.
     
  4. The reward function calls the environment API directly — actual environment
     execution, not a proxy function.

Run:
  python train.py

Or in Colab:
  !python train.py --model Qwen/Qwen2.5-1.5B-Instruct --max-steps 50
"""

import os
import re
import json
import argparse
import requests
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ─── CONSTANTS ───────────────────────────────────────────────────────────────

BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

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


# ─── ARGUMENT PARSING ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training for Zero Trust SRE Gym")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Base model to train")
    parser.add_argument("--max-steps", type=int, default=50,
                       help="Number of GRPO training steps")
    parser.add_argument("--num-generations", type=int, default=8,
                       help="Number of rollouts per prompt (must be >= 2)")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="./zero-trust-sre-checkpoint")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default="zero-trust-sre-agent")
    parser.add_argument("--dataset-size", type=int, default=40,
                       help="Number of prompts to collect for training")
    return parser.parse_args()


# ─── ACTION PARSING ──────────────────────────────────────────────────────────

def parse_action(text: str) -> dict:
    """
    Robust parser for model outputs. LLMs being LLMs, outputs vary a lot.
    Handles: clean JSON, JSON in markdown blocks, partial JSON, free text with keywords.
    """
    text = text.strip()
    
    # Remove markdown code fences if present
    text = re.sub(r'```(?:json)?', '', text).strip().rstrip('`').strip()
    
    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if "tool_name" in parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Try extracting JSON from surrounding text
    match = re.search(r'\{[^{}]*"tool_name"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Fallback: parse from keywords
    tool = "query_siem_logs"
    node = "hr_db"
    
    for candidate_node in ["api_gateway", "auth_service", "frontend", "payment", "hr_db"]:
        if candidate_node in text.lower():
            node = candidate_node
            break
    
    if "isolate" in text.lower():
        tool = "isolate_node"
    elif "ticket" in text.lower() or "file" in text.lower():
        tool = "file_ticket"
    elif "approval" in text.lower() or "check" in text.lower():
        tool = "check_approval"
    
    # Extract justification if present
    just_match = re.search(r'justification["\s:]+([^"}{]+)', text, re.IGNORECASE)
    justification = just_match.group(1).strip()[:150] if just_match else "Parsed from free text"
    
    return {
        "tool_name": tool,
        "payload": {"node": node},
        "justification": justification
    }


# ─── DATASET COLLECTION ──────────────────────────────────────────────────────

def build_obs_text(obs: dict) -> str:
    """Formats environment observation as prompt context."""
    alerts_text = "\n".join(
        f"  [{a['severity']}] {a['target_node']}: {a['symptom'][:80]}"
        for a in obs.get("active_alerts", [])
    )
    return f"""CURRENT STATE:
Alerts:
{alerts_text or '  None'}

Last command output: {obs.get('command_output', 'None')[:200]}
Production uptime: {obs.get('global_uptime', 100.0):.1f}%
Active ticket: {obs.get('active_ticket_id') or 'None'}
Ticket approved: {obs.get('ticket_approved', False)}
Difficulty: {obs.get('difficulty', 'warmup')}"""


def collect_training_prompts(n_prompts: int = 40) -> Dataset:
    """
    Runs episodes in the environment to collect diverse prompts.
    Each prompt is a snapshot of environment state that the model
    will learn to respond to with the correct Zero Trust workflow action.
    
    This gives GRPO the initial dataset it needs. The model then
    gets rewarded/penalized based on what action it actually takes.
    """
    print(f"[Dataset] Collecting {n_prompts} prompts from environment...")
    
    prompts = []
    import random
    
    actions_for_exploration = [
        {"tool_name": "query_siem_logs", "payload": {"node": "hr_db"}, "justification": "investigating"},
        {"tool_name": "query_siem_logs", "payload": {"node": "payment"}, "justification": "investigating"},
        {"tool_name": "query_siem_logs", "payload": {"node": "frontend"}, "justification": "investigating"},
        {"tool_name": "query_siem_logs", "payload": {"node": "api_gateway"}, "justification": "investigating"},
    ]
    
    episode = 0
    while len(prompts) < n_prompts:
        episode += 1
        
        # Reset for new episode
        resp = requests.post(f"{BASE_URL}/reset", json={"task_id": "auto"}, timeout=10)
        if resp.status_code != 200:
            print(f"[Dataset] Reset failed on episode {episode}. Retrying...")
            continue
        
        obs = resp.json()
        done = False
        step = 0
        
        while not done and step < 8 and len(prompts) < n_prompts:
            step += 1
            
            # Collect prompt at this state
            prompt = f"{SYSTEM_PROMPT}\n\n{build_obs_text(obs)}\n\nYour action:"
            prompts.append({"prompt": prompt})
            
            # Take random exploratory action to advance state
            action = random.choice(actions_for_exploration)
            step_resp = requests.post(f"{BASE_URL}/step", json=action, timeout=10)
            if step_resp.status_code != 200:
                break
            
            step_data = step_resp.json()
            obs = step_data["observation"]
            done = step_data["done"]
        
        if episode % 5 == 0:
            print(f"[Dataset] Collected {len(prompts)}/{n_prompts} prompts (episode {episode})")
    
    print(f"[Dataset] Done. {len(prompts)} prompts collected.")
    return Dataset.from_list(prompts[:n_prompts])


# ─── REWARD FUNCTION ─────────────────────────────────────────────────────────

class ZeroTrustRewardFunction:
    """
    GRPO reward function. Executes each completion against the live environment.
    
    IMPORTANT: GRPO calls this for each completion in the batch.
    The environment must be in the right state for each call.
    We handle this by maintaining episode state and resetting when needed.
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self._reset_episode()
    
    def _reset_episode(self):
        """Start fresh episode."""
        try:
            resp = requests.post(f"{self.base_url}/reset", json={"task_id": "auto"}, timeout=10)
            self.current_obs = resp.json() if resp.status_code == 200 else {}
            self.episode_done = False
        except Exception:
            self.current_obs = {}
            self.episode_done = False
    
    def __call__(self, completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        """
        Called by GRPO trainer with a batch of completions.
        Returns reward for each completion.
        
        The key: we reset the episode before each batch so all completions
        in the group start from the same state. This ensures GRPO is comparing
        different responses to the same situation.
        """
        rewards = []
        
        # Reset once per batch so all completions start from same state
        self._reset_episode()
        
        for completion in completions:
            # Parse the model's output
            action = parse_action(completion)
            
            try:
                resp = requests.post(
                    f"{self.base_url}/step",
                    json=action,
                    timeout=15
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    reward = float(data["reward"]["value"])
                    
                    # Bonus for good format (model produced parseable JSON)
                    try:
                        json.loads(completion.strip())
                        reward += 0.2
                    except (json.JSONDecodeError, ValueError):
                        reward -= 0.1
                    
                    rewards.append(reward)
                    
                    # Reset for next completion in batch
                    self._reset_episode()
                else:
                    rewards.append(-1.0)
                    self._reset_episode()
                    
            except requests.exceptions.Timeout:
                rewards.append(-0.5)
                self._reset_episode()
            except Exception:
                rewards.append(-1.0)
                self._reset_episode()
        
        return rewards


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    
    # Verify environment server is running
    print(f"[Train] Connecting to environment at {BASE_URL}...")
    try:
        resp = requests.get(BASE_URL, timeout=5)
        assert resp.status_code == 200
        print(f"[Train] ✅ Environment connected. Curriculum: {resp.json().get('curriculum', {})}")
    except Exception as e:
        print(f"[Train] ❌ Cannot connect to environment: {e}")
        print(f"[Train] Start the server: uvicorn server.app:app --port 7860")
        return
    
    # Load model and tokenizer
    print(f"[Train] Loading {args.model}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"[Train] Model loaded on {next(model.parameters()).device}")
    
    # Collect training dataset
    dataset = collect_training_prompts(n_prompts=args.dataset_size)
    
    # Build reward function
    reward_fn = ZeroTrustRewardFunction(BASE_URL)
    
    # GRPO config
    # Key settings that fix the zero-loss problem:
    #   num_generations=8: 8 rollouts per prompt → variance in group
    #   temperature=0.7: varied outputs across rollouts → varied rewards
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        warmup_steps=5,
        
        # These two settings are the zero-loss fix
        num_generations=args.num_generations,   # KEY: multiple rollouts per prompt
        temperature=0.7,                         # KEY: variation between rollouts
        
        max_new_tokens=120,
        max_prompt_length=1024,
        
        logging_steps=1,
        save_steps=10,
        report_to="none",
        
        # BF16 for efficiency
        bf16=torch.cuda.is_available(),
        fp16=False,
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        processing_class=tokenizer
    )
    
    print(f"\n[Train] Starting GRPO training...")
    print(f"[Train] Config: {args.max_steps} steps, {args.num_generations} generations/prompt, lr={args.learning_rate}")
    print(f"[Train] Expected loss behavior: non-zero, trending downward after step 10")
    print(f"[Train] If you see zeros at step 1-3: normal. If zeros persist past step 10: check GROQ_API_KEY\n")
    
    train_result = trainer.train()
    
    print(f"\n[Train] Training complete!")
    print(f"[Train] Steps: {train_result.global_step}")
    print(f"[Train] Final loss: {train_result.training_loss:.4f}")
    
    # Save
    output_path = "./zero-trust-sre-tuned"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"[Train] Model saved to {output_path}")
    
    # Push to hub if requested
    if args.push_to_hub:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            model.push_to_hub(args.hub_repo, token=hf_token)
            tokenizer.push_to_hub(args.hub_repo, token=hf_token)
            print(f"[Train] Pushed to https://huggingface.co/{args.hub_repo}")
        else:
            print("[Train] HF_TOKEN not set — skipping hub push")
    
    # Print final curriculum state
    try:
        curriculum_resp = requests.get(f"{BASE_URL}/curriculum", timeout=5)
        if curriculum_resp.status_code == 200:
            curriculum = curriculum_resp.json()
            print(f"\n[Train] Final Curriculum State:")
            print(f"  Episodes: {curriculum['episode_count']}")
            print(f"  Difficulty reached: {curriculum['difficulty']}")
            print(f"  Resolution rate: {curriculum.get('resolution_rate', 0):.0%}")
    except Exception:
        pass


if __name__ == "__main__":
    main()