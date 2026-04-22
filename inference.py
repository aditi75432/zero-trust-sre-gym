# inference.py
import os
import time
import requests
import torch
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_llm_output_forgiving(text):
    text = text.lower().strip()
    
    action_identified = "pass"
    target_identified = "none"
    
    valid_actions = [
        "block_ip", "isolate_microservice", "revoke_iam_role",
        "query_siem_logs", "trace_network_ebpf", "inspect_iam_claims",
        "create_ticket", "update_ticket_status", "request_approval", "pass"
    ]
    
    for action in valid_actions:
        if action in text:
            action_identified = action
            break

    target_match = re.search(r"target[\s:\|]+([a-z0-9_\-\.]+)", text)
    if target_match:
        target_identified = target_match.group(1).strip()
        
    return {
        "action_type": action_identified,
        "target": target_identified,
        "justification": "Extracted via forgiving parser"
    }

def run_api_inference():
    print("Loading Tuned Model into Memory...")
    model_path = "./cloudsec-sre-tuned" 
    
    if not os.path.exists(model_path):
        print("ERROR: Tuned model folder not found.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto" if torch.cuda.is_available() else "cpu"
    )

    base_url = "http://localhost:7860"
    task = "hard_insider_threat"

    print(f"\n--- Starting Task: {task} ---")
    
    try:
        response = requests.post(f"{base_url}/reset", json={"task_id": task})
        if response.status_code != 200:
            print("Failed to reset API. Is server running?")
            return
        state_data = response.json()
        obs_json = json.dumps(state_data)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to FastAPI.")
        return

    done = False
    step = 0
    
    while not done and step < 15:
        step += 1
        
        prompt_text = (
            "You are an autonomous Cloud Security SRE operating in a Zero Trust environment.\n"
            "CRITICAL RULES:\n"
            "1. You MUST follow enterprise workflow: Discover Evidence -> Create Ticket -> Request Approval -> Execute Action.\n"
            "2. VALID ACTION TYPES:\n"
            "   - Tools: query_siem_logs, trace_network_ebpf, inspect_iam_claims\n"
            "   - Workflow: create_ticket, update_ticket_status, request_approval\n"
            "   - Remediation: block_ip, isolate_microservice, revoke_iam_role\n"
            "   - pass\n"
            "3. TARGET RULES:\n"
            "   - update_ticket_status target MUST be the Ticket ID (e.g., INC-1234).\n"
            "   - request_approval and block_ip target MUST be the malicious IP or compromised role.\n"
            "   - Tools usually take target 'none'.\n\n"
            "FORMAT YOUR EXACT OUTPUT AS A SINGLE LINE:\n"
            "ACTION: <action_type> | TARGET: <target> | JUSTIFICATION: <brief reason>\n\n"
            f"STATE:\n{obs_json}\n\n"
            "FINAL ANSWER:\n"
        )
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            inputs["input_ids"], attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=50, do_sample=True, temperature=0.2, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        parsed = parse_llm_output_forgiving(response_text)
        
        print(f"Step {step} Agent Action: {parsed['action_type']} on {parsed['target']}")
        
        step_resp = requests.post(f"{base_url}/step", json=parsed)
        if step_resp.status_code == 200:
            step_data = step_resp.json()
            state_data = step_data["observation"]
            obs_json = json.dumps(state_data)
            done = step_data["done"]
            print(f"Feedback: {step_data['reward']['message']} | Reward: {step_data['reward']['value']}")
            time.sleep(2) 
        else:
            break
            
    print("\nInference Complete.")

if __name__ == "__main__":
    run_api_inference()