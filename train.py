import os
import re
import json
import requests
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# --- Configuration ---
ENV_URL = "http://localhost:7860"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct" # Small enough to train fast, smart enough to learn

# --- The System Prompt ---
# This forces the LLM to think before acting and strictly format its JSON.
SYSTEM_PROMPT = """
You are an autonomous Zero Trust SRE Agent.
Your objective is to neutralize threats on the server.
CRITICAL RULE: You CANNOT execute remediation without an approved incident ticket.

You have two tools available:
1. execute_bash (payload requires "command")
2. execute_http_request (payload requires "method", "url", "body")

Your required workflow:
1. Read logs to find the threat IP (do NOT guess).
2. File a Jira ticket via HTTP POST to http://jira-mock:8080/ticket with {"ip": "threat_ip"}.
3. Check approval via HTTP GET to http://jira-mock:8080/ticket/{ticket_id}.
4. Isolate the rogue-service via docker pause.

You must respond in exactly this XML format:
<think>
Your step-by-step reasoning here.
</think>
<action>
{
    "tool_name": "...",
    "payload": {...},
    "justification": "..."
}
</action>
"""

# --- Reward Function 1: Format Compliance ---
def format_reward_func(completions, **kwargs):
    """Rewards the model heavily for using the exact <think> and <action> XML tags."""
    rewards = []
    for comp in completions:
        text = comp[0]['content']
        if "<think>" in text and "</think>" in text and "<action>" in text and "</action>" in text:
            rewards.append(1.0)
        else:
            rewards.append(-2.0)
    return rewards

# --- Reward Function 2: Live Environment Integration ---
def env_reward_func(completions, **kwargs):
    """
    This is the magic. It parses the LLM's action, sends it to your live Zero Trust 
    FastAPI server, and returns the real SLA/Cowboy penalties back to the model.
    """
    rewards = []
    
    # Reset the environment at the start of a batch evaluation
    try:
        requests.post(f"{ENV_URL}/reset", json={"task_id": "training_batch"}, timeout=2)
    except:
        pass # If server is down, we handle it in the loop

    for comp in completions:
        text = comp[0]['content']
        
        # 1. Extract the JSON from the <action> tags
        match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
        if not match:
            rewards.append(-2.0)
            continue
            
        action_json_str = match.group(1).strip()
        
        # 2. Parse the JSON
        try:
            action_dict = json.loads(action_json_str)
        except json.JSONDecodeError:
            rewards.append(-2.0) # Penalty for hallucinating bad JSON
            continue
            
        # 3. Hit the live Zero Trust Environment
        try:
            res = requests.post(f"{ENV_URL}/step", json=action_dict, timeout=3)
            if res.status_code == 200:
                step_data = res.json()
                # Feed the Zero Trust score directly to the RL algorithm
                rewards.append(float(step_data['reward']['value']))
            else:
                rewards.append(-2.0)
        except requests.exceptions.RequestException:
            rewards.append(-2.0) # Server communication error

    return rewards

# --- Dataset Generation ---
# We feed the model the initial alert to trigger the workflow
def create_dataset():
    prompts = []
    for _ in range(50): # 50 episodes
        prompts.append({
            "prompt": "ALERT: SEC-CRITICAL-01. CPU spiking at 99%. Suspected cryptominer or breach on rogue-service. Investigate and neutralize."
        })
    return Dataset.from_list(prompts)

# --- Main Training Loop ---
def main():
    print("Initializing Zero Trust GRPO Pipeline...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = create_dataset()

    # Hackathon Optimized GRPO Settings
    training_args = GRPOConfig(
        output_dir="outputs/zero-trust-qwen",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_prompt_length=512,
        max_completion_length=256,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=25,
        temperature=0.8,
        use_vllm=False # Set to True ONLY if you have a massive GPU (A100) onsite
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, env_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting Live RL Training. Ensure your FastAPI server is running on port 7860.")
    trainer.train()
    
    trainer.save_model("outputs/zero-trust-qwen-final")
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    main()