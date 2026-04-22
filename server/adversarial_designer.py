"""
adversarial_designer.py — Generates targeted incident scenarios using Groq.

This is what makes the environment self-improving. Rather than pulling from
a static pool of scenarios, the designer reads the curriculum's weakness
profile and generates scenarios that specifically target what the agent
fails at. Agent keeps failing at multi-fault? Designer generates more of them.
Agent mastered data exfiltration? Designer stops wasting training time on it.

This directly addresses the root cause of your zero-loss problem:
- Static pool → agent sees same patterns → same reward distribution → zero variance
- LLM designer → novel scenarios each episode → different rewards → real GRPO signal

Has a static fallback pool for when Groq is unavailable, so training
doesn't hard-fail on API issues.
"""

import random
from .llm_client import call_llm_json


# Nodes available in the Zero Trust topology
ALL_NODES = ["api_gateway", "auth_service", "frontend", "payment", "hr_db"]
INTERNAL_NODES = ["frontend", "payment", "hr_db"]   # Can be compromised
GATEWAY_NODES = ["api_gateway", "auth_service"]      # Red herrings only, never compromised


# ─── STATIC FALLBACK POOL ────────────────────────────────────────────────────
# These are used when Groq is unavailable or rate-limited.
# Diverse enough to maintain reward variance for GRPO.

_STATIC_WARMUP = [
    {
        "compromised_nodes": ["hr_db"],
        "threat_type": "data_exfiltration",
        "threat_ips": ["10.0.5.42"],
        "iam_roles": ["hr-reader-svc"],
        "red_herring_nodes": ["api_gateway"],
        "siem_evidence_template": "Unauthorized IAM role assumption detected. Source IP {ip}. 287MB outbound to external endpoint. IAM role {role} assumed outside business hours.",
        "difficulty_description": "Single exfiltration from HR database"
    },
    {
        "compromised_nodes": ["payment"],
        "threat_type": "lateral_movement",
        "threat_ips": ["10.0.3.17"],
        "iam_roles": ["payment-processor-svc"],
        "red_herring_nodes": ["frontend"],
        "siem_evidence_template": "Payment service probing internal auth endpoints. Source IP {ip}. IAM role {role} querying /admin/users 47 times. Not in normal traffic pattern.",
        "difficulty_description": "Lateral movement from payment to auth layer"
    },
    {
        "compromised_nodes": ["frontend"],
        "threat_type": "privilege_escalation",
        "threat_ips": ["10.0.7.88"],
        "iam_roles": ["frontend-webapp-svc"],
        "red_herring_nodes": ["auth_service"],
        "siem_evidence_template": "Frontend service attempting to assume admin IAM role. Source IP {ip}. Role {role} made 12 AssumeRole calls targeting admin-full-access. All denied but attempts continuing.",
        "difficulty_description": "Frontend attempting privilege escalation to admin"
    }
]

_STATIC_ADVANCED = [
    {
        "compromised_nodes": ["payment", "hr_db"],
        "threat_type": "data_exfiltration",
        "threat_ips": ["10.0.5.42", "10.0.3.77"],
        "iam_roles": ["payment-processor-svc", "hr-reader-svc"],
        "red_herring_nodes": ["api_gateway"],
        "siem_evidence_template": "Simultaneous compromise detected. Node {node}: Source IP {ip}, role {role}. Coordinated attack pattern — both services exfiltrating to same external endpoint.",
        "difficulty_description": "Coordinated two-node attack across payment and HR systems"
    },
    {
        "compromised_nodes": ["frontend", "payment"],
        "threat_type": "lateral_movement",
        "threat_ips": ["10.0.8.12", "10.0.8.13"],
        "iam_roles": ["frontend-webapp-svc", "payment-processor-svc"],
        "red_herring_nodes": ["auth_service"],
        "siem_evidence_template": "Cascading lateral movement: frontend ({ip}) pivoted to payment node. Both roles ({role}) now active. Sequential access pattern detected.",
        "difficulty_description": "Lateral movement has spread from frontend to payment"
    }
]


# ─── MAIN API ────────────────────────────────────────────────────────────────

def generate_scenario(weakness_profile: dict, difficulty: str = "intermediate") -> dict:
    """
    Main entry point. Tries LLM generation, falls back to static pool.
    """
    try:
        scenario = _llm_generate(weakness_profile, difficulty)
        # Validate the LLM gave us something usable
        if scenario.get("compromised_nodes") and scenario.get("threat_ips"):
            return _normalize(scenario, difficulty) # <-- FIX: Passing difficulty
    except Exception as e:
        print(f"[AdversarialDesigner] LLM generation failed ({type(e).__name__}: {e}). Using static pool.")
    
    return _normalize(_static_pick(difficulty), difficulty)


def _llm_generate(weakness_profile: dict, difficulty: str) -> dict:
    """
    Asks Groq to design a scenario that targets the agent's weaknesses.
    The model has full creative latitude within the topology constraints.
    """
    # Find what the agent struggles with most
    top_weaknesses = sorted(
        weakness_profile.items(), key=lambda x: x[1], reverse=True
    )[:2]
    weakness_focus = ", ".join(f"{k} (failure rate: {v:.0%})" for k, v in top_weaknesses) or "any threat type"
    
    is_multi_fault = difficulty in ["advanced", "expert"]
    node_count = 2 if is_multi_fault else 1
    
    prompt = f"""You are a Red Team adversarial scenario designer for a Zero Trust enterprise security training system.

System topology:
- Nodes: api_gateway, auth_service, frontend, payment, hr_db
- IMPORTANT: api_gateway and auth_service can NEVER be compromised (they are red herrings only)
- Only these can be compromised: frontend, payment, hr_db

Agent weakness profile (higher = agent fails at this):
{weakness_profile}
Focus areas: {weakness_focus}

Configuration:
- Difficulty: {difficulty}
- Number of compromised nodes: {node_count}
- Multi-fault: {is_multi_fault}

Design a realistic enterprise security incident. Requirements:
1. compromised_nodes: list of {node_count} node(s) from [frontend, payment, hr_db] ONLY
2. threat_ips: one internal IP per compromised node (10.0.x.x format)  
3. iam_roles: one realistic service account name per compromised node
4. red_herring_nodes: 1-2 nodes that will have misleading WARNING alerts (must differ from compromised)
5. siem_evidence_template: what SIEM shows when agent queries the compromised node. Use {{ip}} and {{role}} as placeholders.
6. threat_type: one of data_exfiltration, lateral_movement, privilege_escalation, supply_chain

Make it target the agent's weaknesses. Be creative with the attack narrative.

Respond ONLY with valid JSON, no surrounding text:
{{
  "compromised_nodes": ["node1"],
  "threat_type": "data_exfiltration",
  "threat_ips": ["10.0.x.x"],
  "iam_roles": ["role-name-svc"],
  "red_herring_nodes": ["node"],
  "siem_evidence_template": "Evidence text with {{ip}} and {{role}} placeholders",
  "difficulty_description": "One sentence describing what makes this scenario challenging"
}}"""

    return call_llm_json(prompt, temperature=0.7, model="llama-3.1-8b-instant")


def _static_pick(difficulty: str) -> dict:
    """Deterministic fallback — always returns a valid scenario."""
    if difficulty in ["advanced", "expert"]:
        return random.choice(_STATIC_ADVANCED)
    return random.choice(_STATIC_WARMUP)


def _normalize(raw: dict, difficulty: str = "unknown") -> dict: # <-- FIX: Accepting difficulty
    """
    Ensures the scenario dict has all required keys with valid values.
    Prevents KeyError crashes in environment.py from partial LLM output.
    """
    # Ensure compromised nodes are all valid internal nodes
    raw_nodes = raw.get("compromised_nodes", ["hr_db"])
    valid_nodes = [n for n in raw_nodes if n in INTERNAL_NODES]
    if not valid_nodes:
        valid_nodes = ["hr_db"]
    
    # Ensure IPs are present for each compromised node
    raw_ips = raw.get("threat_ips", [])
    while len(raw_ips) < len(valid_nodes):
        raw_ips.append(f"10.0.{random.randint(1, 10)}.{random.randint(10, 99)}")
    
    # Ensure IAM roles present
    raw_roles = raw.get("iam_roles", [])
    role_defaults = ["frontend-webapp-svc", "payment-processor-svc", "hr-reader-svc"]
    while len(raw_roles) < len(valid_nodes):
        raw_roles.append(role_defaults[len(raw_roles) % len(role_defaults)])
    
    # Ensure red herring nodes don't overlap with compromised
    raw_rh = raw.get("red_herring_nodes", [])
    red_herrings = [n for n in raw_rh if n not in valid_nodes and n in ALL_NODES]
    if not red_herrings:
        candidates = [n for n in GATEWAY_NODES if n not in valid_nodes]
        red_herrings = candidates[:1]
    
    return {
        "compromised_nodes": valid_nodes,
        "threat_type": raw.get("threat_type", "data_exfiltration"),
        "threat_ips": raw_ips[:len(valid_nodes)],
        "iam_roles": raw_roles[:len(valid_nodes)],
        "red_herring_nodes": red_herrings,
        "siem_evidence_template": raw.get(
            "siem_evidence_template",
            "Unauthorized IAM role assumption. Source IP {ip}. Role {role} active outside policy. Data transfer anomaly detected."
        ),
        "difficulty_description": raw.get("difficulty_description", f"{difficulty} security incident")
    }