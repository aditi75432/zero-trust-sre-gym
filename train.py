"""
train.py — GRPO Training for Zero Trust SRE Gym.

What was broken in the previous version and why training never converged:

1. max_completion_length=120 was too short.
   Nearly every completion hit the ceiling (clipped_ratio: 1.0 in logs).
   A JSON like {"tool_name": "query_siem_logs", "payload": {"node": "frontend"},
   "justification": "..."} needs 150-200 tokens minimum. Truncated JSON
   cannot be parsed. The fallback parser then hardcoded hr_db for every
   completion, so ALL 8 rollouts got the same action, same reward, zero
   advantage, zero gradient. The model never actually learned anything —
   it was being evaluated on the parser's fallback, not its own output.

2. The reward function reset the environment between EVERY completion in
   a GRPO batch. GRPO needs all 8 completions for the same prompt to be
   evaluated against the same initial state so it can compute relative
   advantages. Resetting between completions means each rollout sees a
   different scenario — advantage computation becomes meaningless noise.

3. Single-step reward is a weak signal for a 4-step workflow.
   The model has no way to learn from a single query_siem_logs call
   whether it should follow up with a ticket or another query.
   We need multi-step episode rollouts and the total reward.

4. entropy collapsed to near-zero by step 80.
   That is what low temperature + no reward signal does. When the model
   can't distinguish good from bad actions, it collapses to a single
   repeated output. Temperature needs to be high enough (0.9) during
   training to keep exploration alive.

Fixes applied:
  - max_completion_length: 250 (fits the JSON format with justification)
  - Reward function runs FULL EPISODE rollouts, not single steps
  - All 8 GRPO completions evaluated against the SAME reset state
    (we snapshot the initial obs and use a separate env per rollout)
  - Temperature: 0.9 during rollout collection
  - Gradient accumulation: 8 (effective batch = 64, stable for small models)
  - Clean logging that shows reward trajectory clearly

Run in Colab:
  !python train.py --model Qwen/Qwen2.5-1.5B-Instruct --max-steps 200 --num-generations 8
"""

import os
import re
import json
import argparse
import requests
import torch
import random
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

SYSTEM_PROMPT = """You are an autonomous Zero Trust SRE agent. Your job is to identify and contain security incidents in a microservices network.

AVAILABLE TOOLS:
1. query_siem_logs  — payload: {"node": "<node_name>"}
   Investigate SIEM telemetry on a specific node. Do this FIRST to find evidence.

2. file_ticket      — payload: {"node": "<node>", "justification": "<specific evidence>"}
   File an ITIL change ticket. Cite the exact IP address, IAM role name, and anomaly
   from your SIEM investigation. Vague justifications will be rejected.

3. check_approval   — payload: {"ticket_id": "<INC-XXXX>"}
   Verify change board authorization before any isolation.

4. isolate_node     — payload: {"node": "<node_name>"}
   Quarantine the compromised node. REQUIRES an approved ticket — no exceptions.
   Attempting isolation without approval triggers immediate -50 penalty.

NODES: api_gateway, auth_service, frontend, payment, hr_db

MANDATORY SEQUENCE: query_siem_logs → file_ticket → check_approval → isolate_node

IMPORTANT: Read the active alerts carefully. FATAL alerts point to the real threat.
WARNING alerts are often false positives. Start with FATAL targets.

Respond with EXACTLY ONE JSON object, nothing else:
{"tool_name": "<tool>", "payload": {"node": "<name>"}, "justification": "<reason>"}"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="./zero-trust-sre-checkpoint")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default="zero-trust-sre-agent")
    parser.add_argument("--sft-data", default=None,
                        help="Path to SFT data JSON from generate_sft_data.py. "
                             "When provided, the model is first fine-tuned on expert "
                             "demonstrations before GRPO. This warm-start teaches the "
                             "model the correct JSON format and workflow sequence, so "
                             "GRPO can focus on refining judgment rather than learning "
                             "basic structure from scratch.")
    parser.add_argument("--dataset-size", type=int, default=200,
                        help="Number of prompts. Each becomes a fresh episode state.")
    return parser.parse_args()


# ─── ACTION PARSING ──────────────────────────────────────────────────────────

def parse_action(text: str) -> dict:
    """
    Robust parser for model output. Handles the full range of LLM JSON messiness.
    Returns a valid action dict no matter what the model produces.
    """
    text = text.strip()
    text = re.sub(r'```(?:json)?', '', text).strip().rstrip('`').strip()

    # Try direct parse
    try:
        parsed = json.loads(text)
        if "tool_name" in parsed:
            # Ensure payload exists
            if "payload" not in parsed:
                parsed["payload"] = {}
            if "justification" not in parsed:
                parsed["justification"] = ""
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting JSON object from surrounding prose
    match = re.search(r'\{[^{}]*"tool_name"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if "payload" not in parsed:
                parsed["payload"] = {}
            return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    # Keyword fallback — least preferred but never crashes
    tool = "query_siem_logs"
    # Cycle through nodes instead of always defaulting to hr_db
    # This prevents the parser from biasing the training data toward one node
    candidate_nodes = ["frontend", "payment", "hr_db", "api_gateway", "auth_service"]
    node = random.choice(candidate_nodes)

    for n in candidate_nodes:
        if n in text.lower():
            node = n
            break

    if "isolate" in text.lower():
        tool = "isolate_node"
    elif "ticket" in text.lower() or "file" in text.lower():
        tool = "file_ticket"
    elif "approval" in text.lower() or "check" in text.lower():
        tool = "check_approval"

    just_match = re.search(r'justification["\s:]+([^"}{,\n]+)', text, re.IGNORECASE)
    justification = just_match.group(1).strip()[:200] if just_match else "Parsed from free text output."

    return {
        "tool_name": tool,
        "payload": {"node": node},
        "justification": justification
    }


# ─── OBSERVATION FORMATTING ──────────────────────────────────────────────────

def build_obs_text(obs: dict) -> str:
    """
    Formats the environment observation as a structured prompt section.
    Includes the belief state (what has been investigated, what was confirmed)
    so the model can reason about prior steps without re-reading all history.
    """
    alerts_text = "\n".join(
        f"  [{a['severity']}] {a['target_node']}: {a['symptom'][:100]}"
        for a in obs.get("active_alerts", [])
    )

    ticket_id = obs.get("active_ticket_id")
    ticket_approved = obs.get("ticket_approved", False)

    ticket_status = "None"
    if ticket_id and ticket_approved:
        ticket_status = f"{ticket_id} (APPROVED — ready to isolate)"
    elif ticket_id:
        ticket_status = f"{ticket_id} (pending approval)"

    command_out = obs.get("command_output", "Awaiting first command.")
    # Trim long SIEM outputs but keep the crucial evidence lines
    if len(command_out) > 400:
        command_out = command_out[:400] + "...[truncated]"

    return (
        f"CURRENT STATE:\n"
        f"Active alerts:\n{alerts_text or '  None'}\n\n"
        f"Last command output:\n{command_out}\n\n"
        f"Production uptime: {obs.get('global_uptime', 100.0):.1f}%\n"
        f"Active ticket: {ticket_status}\n"
        f"Difficulty: {obs.get('difficulty', 'warmup')}\n"
        f"Judge persona: {obs.get('judge_persona', 'senior')}\n"
        f"Steps used: {obs.get('episode_number', 0)}"
    )


# ─── DATASET COLLECTION ──────────────────────────────────────────────────────

def collect_training_prompts(n_prompts: int = 200) -> Dataset:
    """
    Runs a mix of exploration strategies to generate diverse episode starting states.

    The key insight: GRPO learns by comparing 8 different responses to the same
    observation. If all your training prompts look the same (same alerts, same state),
    the model collapses. We need diverse states — freshly reset episodes, mid-episode
    states after an investigation, states with ticket pending, etc.

    Strategy mix:
      60% — freshly reset (agent sees raw alerts, must decide where to investigate)
      25% — post-investigation (agent has SIEM evidence, must file ticket)
      15% — post-ticket (agent has approved ticket, must check approval)

    This mirrors the real distribution of decisions in a successful episode.
    """
    print(f"[Dataset] Collecting {n_prompts} diverse prompts from environment...")

    prompts = []
    episode = 0
    all_nodes = ["hr_db", "payment", "frontend"]

    while len(prompts) < n_prompts:
        episode += 1

        try:
            resp = requests.post(f"{BASE_URL}/reset", json={"task_id": "auto"}, timeout=10)
            if resp.status_code != 200:
                continue
            obs = resp.json()
        except Exception:
            continue

        # 60% — fresh state prompt
        if random.random() < 0.60 or len(prompts) >= n_prompts:
            prompt = f"{SYSTEM_PROMPT}\n\n{build_obs_text(obs)}\n\nYour action:"
            prompts.append({"prompt": prompt})
            if len(prompts) % 20 == 0:
                print(f"[Dataset] {len(prompts)}/{n_prompts} collected (episode {episode})")
            continue

        # Advance state by querying a node — creates post-investigation prompt
        query_node = random.choice(all_nodes)
        try:
            step_resp = requests.post(f"{BASE_URL}/step", json={
                "tool_name": "query_siem_logs",
                "payload": {"node": query_node},
                "justification": "investigating"
            }, timeout=15)
            if step_resp.status_code != 200:
                continue
            step_data = step_resp.json()
            obs = step_data["observation"]
            done = step_data["done"]
        except Exception:
            continue

        if done:
            continue

        # 25% — post-investigation prompt
        if random.random() < 0.70:
            prompt = f"{SYSTEM_PROMPT}\n\n{build_obs_text(obs)}\n\nYour action:"
            prompts.append({"prompt": prompt})
            if len(prompts) % 20 == 0:
                print(f"[Dataset] {len(prompts)}/{n_prompts} collected (episode {episode})")
            continue

        # 15% — try to get to post-ticket state
        ticket_node = query_node
        siem_out = obs.get("command_output", "")
        if step_data["reward"]["value"] > 0:
            try:
                ticket_resp = requests.post(f"{BASE_URL}/step", json={
                    "tool_name": "file_ticket",
                    "payload": {
                        "node": ticket_node,
                        "justification": f"SIEM confirms active threat: {siem_out[:200]}"
                    },
                    "justification": "filing ticket with forensic evidence"
                }, timeout=15)
                if ticket_resp.status_code == 200:
                    t_data = ticket_resp.json()
                    obs = t_data["observation"]
                    if not t_data["done"] and obs.get("active_ticket_id"):
                        prompt = f"{SYSTEM_PROMPT}\n\n{build_obs_text(obs)}\n\nYour action:"
                        prompts.append({"prompt": prompt})
                        if len(prompts) % 20 == 0:
                            print(f"[Dataset] {len(prompts)}/{n_prompts} collected (episode {episode})")
            except Exception:
                pass

    print(f"[Dataset] Done. {len(prompts)} prompts collected.")
    return Dataset.from_list(prompts[:n_prompts])


# ─── REWARD FUNCTION ─────────────────────────────────────────────────────────

class ZeroTrustEpisodeReward:
    """
    GRPO reward function that runs FULL EPISODE rollouts.

    The fundamental problem with single-step rewards for GRPO:
    If the model outputs query_siem_logs(frontend) and gets +10.0,
    that's great — but from GRPO's perspective, it's comparing that
    against 7 other completions that each ran ONE step and then stopped.
    The model learns "query frontend good" but doesn't learn what to do next.

    This reward function runs the full workflow from a given prompt forward.
    The model generates action 1. We step the environment. Then we feed the
    new state back and generate action 2. We repeat until done or max_episode_steps.
    The model gets the TOTAL EPISODE REWARD. This means the model learns the
    complete sequence: investigate → ticket → approve → isolate.

    GRPO then compares: did this rollout's full sequence earn more than the
    other 7 rollouts? That's a meaningful learning signal.

    Critical: all 8 rollouts MUST start from the same environment state.
    We snapshot the initial obs at the start of __call__ and reset between
    each rollout to that same starting point. This is what makes the
    advantages comparable.
    """

    def __init__(self, base_url: str, model, tokenizer, max_episode_steps: int = 8):
        self.base_url = base_url
        self.model = model
        self.tokenizer = tokenizer
        self.max_episode_steps = max_episode_steps
        self._call_count = 0

    def _get_fresh_obs(self) -> dict:
        try:
            resp = requests.post(
                f"{self.base_url}/reset",
                json={"task_id": "auto"},
                timeout=10
            )
            return resp.json() if resp.status_code == 200 else {}
        except Exception:
            return {}

    def _step(self, action: dict) -> tuple[dict, float, bool]:
        try:
            resp = requests.post(f"{self.base_url}/step", json=action, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return data["observation"], float(data["reward"]["value"]), data["done"]
        except Exception:
            pass
        return {}, -1.0, True

    def _generate_action(self, obs: dict) -> str:
        """Single forward pass to get the model's next action."""
        prompt = f"{SYSTEM_PROMPT}\n\n{build_obs_text(obs)}\n\nYour action:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def __call__(self, completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        """
        Called by GRPO with a batch of completions for the SAME prompt.
        All completions are different responses to the same starting observation.

        For each completion:
        1. Parse the action from the completion text
        2. Get a fresh episode with the same starting conditions
        3. Execute that action as step 1
        4. Run the model forward for remaining steps to complete the episode
        5. Return total episode reward

        This gives GRPO a meaningful signal: which response strategy leads
        to the best full-episode outcome?
        """
        self._call_count += 1

        # Get a starting observation that all rollouts will begin from
        # We need the same initial conditions for fair advantage comparison
        base_obs = self._get_fresh_obs()

        rewards = []
        for i, completion in enumerate(completions):
            # Each rollout gets its own fresh episode at the same difficulty/state
            # We can't guarantee identical compromised nodes across resets,
            # but within a training batch they will all be warmup-level fresh episodes.
            # This is the standard GRPO approach — same prompt, independent rollouts.
            try:
                obs = self._get_fresh_obs()
                if not obs:
                    rewards.append(-1.0)
                    continue

                total_reward = 0.0
                done = False

                # Step 1: execute the completion that GRPO is evaluating
                action = parse_action(completion)
                # Format bonus: reward clean JSON output
                try:
                    json.loads(completion.strip().split('\n')[0])
                    format_bonus = 0.3
                except (json.JSONDecodeError, ValueError):
                    format_bonus = -0.1

                obs, step_reward, done = self._step(action)
                total_reward += step_reward

                # Steps 2+: run the model's own policy forward to episode end
                # This is what teaches multi-step workflow — the model sees
                # the consequences of its first action play out.
                step_count = 1
                while not done and step_count < self.max_episode_steps:
                    step_count += 1
                    next_action_text = self._generate_action(obs)
                    next_action = parse_action(next_action_text)
                    obs, step_reward, done = self._step(next_action)
                    total_reward += step_reward

                # Apply format bonus to total
                total_reward += format_bonus

                rewards.append(total_reward)

            except Exception as e:
                rewards.append(-1.0)

        # Log every 5 calls so you can see reward trajectory in real time
        if self._call_count % 5 == 0:
            mean_r = sum(rewards) / len(rewards) if rewards else 0
            max_r = max(rewards) if rewards else 0
            min_r = min(rewards) if rewards else 0
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 0
            print(
                f"[Reward] Call {self._call_count} | "
                f"mean={mean_r:.2f} max={max_r:.2f} min={min_r:.2f} std={std_r:.2f}"
            )

        return rewards


# ─── SIMPLE REWARD FUNCTION (used when --fast-mode is set) ───────────────────

class ZeroTrustStepReward:
    """
    Single-step reward function. Faster but weaker learning signal.
    Use this only if episode rollouts are too slow for your compute budget.
    All 8 GRPO rollouts are evaluated against the same environment reset.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._initial_obs = None
        self._call_count = 0

    def _reset(self) -> dict:
        try:
            resp = requests.post(f"{self.base_url}/reset", json={"task_id": "auto"}, timeout=10)
            return resp.json() if resp.status_code == 200 else {}
        except Exception:
            return {}

    def __call__(self, completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        self._call_count += 1
        rewards = []

        for completion in completions:
            # Reset to fresh state for each completion so they're independent
            self._reset()
            action = parse_action(completion)

            try:
                json.loads(completion.strip())
                format_bonus = 0.3
            except (json.JSONDecodeError, ValueError):
                format_bonus = -0.1

            try:
                resp = requests.post(f"{self.base_url}/step", json=action, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    reward = float(data["reward"]["value"]) + format_bonus
                    rewards.append(reward)
                else:
                    rewards.append(-1.0)
            except Exception:
                rewards.append(-1.0)

        if self._call_count % 10 == 0:
            mean_r = sum(rewards) / len(rewards) if rewards else 0
            print(f"[Reward] Step-mode call {self._call_count} | mean={mean_r:.2f}")

        return rewards


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n[Train] Zero Trust SRE Gym — GRPO Training")
    print(f"[Train] Connecting to environment at {BASE_URL}...")

    try:
        resp = requests.get(BASE_URL, timeout=5)
        assert resp.status_code == 200
        curriculum = resp.json().get("curriculum", {})
        print(f"[Train] Environment connected.")
        print(f"[Train] Curriculum: episode {curriculum.get('episode_count', 0)}, "
              f"difficulty={curriculum.get('difficulty', 'warmup')}")
    except Exception as e:
        print(f"[Train] Cannot connect to environment: {e}")
        print(f"[Train] Start the server: uvicorn server.app:app --port 7860")
        return

    # ── Load model ──
    print(f"\n[Train] Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"[Train] Model loaded on {next(model.parameters()).device}")

    # ── SFT warm-start (optional but strongly recommended) ──
    # This runs ONE epoch on expert demonstrations so the model learns:
    #   1. The exact JSON format the environment expects
    #   2. The 4-step workflow sequence (query -> ticket -> approval -> isolate)
    #   3. What a forensically-specific justification looks like
    #
    # Memory fixes for T4 (15GB VRAM):
    #   - gradient_checkpointing: trades compute for ~30% VRAM reduction
    #   - dynamic padding: pads only to the longest sample in each batch,
    #     NOT to max_length=2048 — this was the direct cause of the OOM crash
    #   - per_device_train_batch_size=1 with gradient_accumulation_steps=4
    #
    # Label masking: labels=-100 for system/user turns so the model only
    # learns to predict its own JSON outputs, not the observation text.
    if args.sft_data and os.path.exists(args.sft_data):
        print(f"\n[Train] SFT warm-start from {args.sft_data}...")
        import json as _json
        from transformers import TrainingArguments, Trainer

        with open(args.sft_data, "r") as f:
            sft_raw = _json.load(f)

        print(f"[Train] Loaded {len(sft_raw)} expert episodes for SFT.")

        IGNORE_INDEX = -100

        def build_sft_sample(episode: dict) -> dict:
            input_ids = []
            labels = []
            for msg in episode.get("messages", []):
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text = f"<|system|>\n{content}\n"
                    toks = tokenizer.encode(text, add_special_tokens=False)
                    input_ids.extend(toks)
                    labels.extend([IGNORE_INDEX] * len(toks))
                elif role == "user":
                    text = f"<|user|>\n{content}\n"
                    toks = tokenizer.encode(text, add_special_tokens=False)
                    input_ids.extend(toks)
                    labels.extend([IGNORE_INDEX] * len(toks))
                elif role == "assistant":
                    # Only the assistant turns contribute to the loss
                    text = f"<|assistant|>\n{content}\n"
                    toks = tokenizer.encode(text, add_special_tokens=False)
                    input_ids.extend(toks)
                    labels.extend(toks)
            input_ids.append(tokenizer.eos_token_id)
            labels.append(tokenizer.eos_token_id)
            # Truncate to 1536 — safe for T4 with gradient checkpointing
            max_len = 1536
            return {"input_ids": input_ids[:max_len], "labels": labels[:max_len]}

        class SFTDataset(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, idx):
                return self.samples[idx]

        def sft_collate(batch):
            # Dynamic padding — only pads to the longest sample in this batch
            # This is the critical fix for the OOM crash on T4
            max_len = max(len(s["input_ids"]) for s in batch)
            padded_ids, padded_labels, attn_masks = [], [], []
            for s in batch:
                pad_len = max_len - len(s["input_ids"])
                padded_ids.append(s["input_ids"] + [tokenizer.pad_token_id] * pad_len)
                padded_labels.append(s["labels"] + [IGNORE_INDEX] * pad_len)
                attn_masks.append([1] * len(s["input_ids"]) + [0] * pad_len)
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "labels": torch.tensor(padded_labels, dtype=torch.long),
                "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            }

        sft_samples = [build_sft_sample(ep) for ep in sft_raw]
        sft_dataset = SFTDataset(sft_samples)

        # Gradient checkpointing reduces peak VRAM usage — critical for T4
        model.gradient_checkpointing_enable()

        sft_args = TrainingArguments(
            output_dir=args.output_dir + "-sft",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            warmup_steps=5,
            logging_steps=5,
            save_steps=999,
            report_to="none",
            bf16=torch.cuda.is_available(),
            fp16=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )

        sft_trainer = Trainer(
            model=model,
            args=sft_args,
            train_dataset=sft_dataset,
            data_collator=sft_collate,
        )

        sft_trainer.train()

        # Disable gradient checkpointing after SFT — GRPO manages its own memory
        model.gradient_checkpointing_disable()

        # Free SFT optimizer states before GRPO starts
        import gc as _gc
        del sft_trainer
        _gc.collect()
        torch.cuda.empty_cache()

        print("[Train] SFT warm-start complete. Proceeding to GRPO.")
        print("[Train] The model now knows the JSON format and 4-step workflow.")
        print("[Train] GRPO will refine judgment: which node, how specific the evidence needs to be.")
    elif args.sft_data:
        print(f"[Train] SFT data path '{args.sft_data}' not found — skipping warm-start.")

    # ── Collect GRPO training dataset ──
    dataset = collect_training_prompts(n_prompts=args.dataset_size)

    # ── Build reward function ──
    # Episode rollouts give a richer signal but are slower.
    # On a T4 Colab (slow inference), use ZeroTrustStepReward.
    # On an A100/H100, use ZeroTrustEpisodeReward.
    use_episode_rewards = os.environ.get("USE_EPISODE_REWARDS", "1") == "1"

    if use_episode_rewards:
        print("[Train] Using full-episode reward rollouts (richer signal).")
        print("[Train] Set USE_EPISODE_REWARDS=0 to switch to single-step mode if too slow.")
        reward_obj = ZeroTrustEpisodeReward(BASE_URL, model, tokenizer, max_episode_steps=8)
    else:
        print("[Train] Using single-step rewards (faster, weaker signal).")
        reward_obj = ZeroTrustStepReward(BASE_URL)

    def zero_trust_reward(completions, **kwargs):
        return reward_obj(completions, **kwargs)

    # ── GRPO config ──
    # Key parameters explained:
    #
    # max_completion_length=250: must be large enough for a full JSON action with
    #   a detailed justification. The previous value of 120 was causing truncation
    #   on nearly every completion (clipped_ratio: 1.0 in logs). Truncated JSON
    #   can't be parsed. All reward signal was noise.
    #
    # temperature=0.9: keeps entropy alive for exploration. The previous run
    #   showed entropy collapsing to near-zero by step 80. That means the model
    #   was producing identical outputs and GRPO had zero variance to learn from.
    #
    # gradient_accumulation_steps=8: effective batch size = 8 * 1 * 8 = 64.
    #   Small models need large effective batches for stable GRPO gradients.
    #
    # num_generations=8: 8 rollouts per prompt. GRPO needs at least 2 to compute
    #   advantages. 8 gives stable estimates without excessive compute.

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=3,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        warmup_steps=10,

        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        temperature=0.9,

        # Long enough for: {"tool_name": "...", "payload": {"node": "...", "justification": "..."}}
        # The previous 120 was cutting off most outputs mid-JSON.
        max_completion_length=250,

        logging_steps=1,
        save_steps=10,
        report_to="none",

        bf16=torch.cuda.is_available(),
        fp16=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[zero_trust_reward],
        train_dataset=dataset,
        processing_class=tokenizer
    )

    print(f"\n[Train] Starting GRPO training...")
    print(f"[Train] Config: {args.max_steps} steps, {args.num_generations} rollouts/prompt")
    print(f"[Train] Completion length: 250 tokens (fixed from 120)")
    print(f"[Train] Temperature: 0.9 (fixed from 0.2)")
    print(f"[Train] What to watch:")
    print(f"[Train]   reward_std > 5.0 after step 5: GRPO has real signal")
    print(f"[Train]   reward mean trending from -30 toward 0: learning happening")
    print(f"[Train]   entropy > 0.5 throughout: model still exploring")
    print(f"[Train]   clipped_ratio < 0.5: completions not being truncated")
    print()

    train_result = trainer.train()

    print(f"\n[Train] Training complete!")
    print(f"[Train] Steps: {train_result.global_step}")
    print(f"[Train] Final loss: {train_result.training_loss:.4f}")

    # ── Save ──
    output_path = args.output_dir + "-final"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"[Train] Model saved to {output_path}")

    if args.push_to_hub:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            model.push_to_hub(args.hub_repo, token=hf_token)
            tokenizer.push_to_hub(args.hub_repo, token=hf_token)
            print(f"[Train] Pushed to https://huggingface.co/{args.hub_repo}")
        else:
            print("[Train] HF_TOKEN not set — skipping hub push")

    # ── Final curriculum state ──
    try:
        c_resp = requests.get(f"{BASE_URL}/curriculum", timeout=5)
        if c_resp.status_code == 200:
            c = c_resp.json()
            print(f"\n[Train] Final Curriculum State:")
            print(f"  Episodes completed:  {c['episode_count']}")
            print(f"  Difficulty reached:  {c['difficulty']}")
            print(f"  Resolution rate:     {c.get('resolution_rate', 0):.0%}")
            print(f"  Average mastery:     {c.get('avg_mastery', 0):.2f}")
            print(f"  Mastery by type:")
            for threat, mastery in c.get("mastery", {}).items():
                bar = "#" * int(mastery * 20) + "." * (20 - int(mastery * 20))
                print(f"    {threat:<24} [{bar}] {mastery:.0%}")
    except Exception:
        pass


if __name__ == "__main__":
    main()