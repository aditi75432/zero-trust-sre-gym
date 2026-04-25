import random
import requests
import time
from .llm_client import call_llm_json

def fetch_live_threat_intel() -> str:
    url = "https://cve.circl.lu/api/last"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                cves = r.json()
                for cve in cves:
                    if cve.get('cvss', 0) >= 7.5:
                        return f"Real Threat Intel: {cve['id']} - {cve['summary']}"
            print(f"[ThreatIntel] Attempt {attempt+1}: status {r.status_code}")
        except Exception as e:
            print(f"[ThreatIntel] Attempt {attempt+1} failed: {e}")
        time.sleep(2)
    return "Real Threat Intel: CVE-2024-3400 (Palo Alto PAN-OS Command Injection allowing unauthenticated RCE)."

ALL_NODES = ["api_gateway", "auth_service", "frontend", "payment", "hr_db"]
INTERNAL_NODES = ["frontend", "payment", "hr_db"]
GATEWAY_NODES = ["api_gateway", "auth_service"]

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
        "siem_evidence_template": "Simultaneous compromise detected. Node {node}: Source IP {ip}, role {role}. Coordinated attack pattern -- both services exfiltrating to same external endpoint.",
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


def generate_scenario(weakness_profile: dict, difficulty: str = "intermediate") -> dict:
    try:
        scenario = _llm_generate(weakness_profile, difficulty)
        if scenario.get("compromised_nodes") and scenario.get("threat_ips"):
            return _normalize(scenario, difficulty)
    except Exception as e:
        print(f"[AdversarialDesigner] LLM generation failed ({type(e).__name__}: {e}). Using static pool.")
    
    return _normalize(_static_pick(difficulty), difficulty)


def _llm_generate(weakness_profile: dict, difficulty: str) -> dict:
    top_weaknesses = sorted(weakness_profile.items(), key=lambda x: x[1], reverse=True)[:2]
    weakness_focus = ", ".join(f"{k} (failure rate: {v:.0%})" for k, v in top_weaknesses) or "any threat type"
    
    is_multi_fault = difficulty in ["advanced", "expert"]
    node_count = 2 if is_multi_fault else 1
    
    live_cve_context = fetch_live_threat_intel()
    
    prompt = f"""You are a Red Team adversarial scenario designer for a Zero Trust enterprise security system.

System topology:
- Nodes: api_gateway, auth_service, frontend, payment, hr_db
- Red herrings: api_gateway, auth_service (NEVER compromised)
- Vulnerable targets: frontend, payment, hr_db

Agent weaknesses: {weakness_focus}
Difficulty: {difficulty} ({node_count} nodes compromised)

CRITICAL REAL-WORLD INJECTION:
You MUST base the SIEM evidence on this exact live vulnerability:
>>> {live_cve_context} <<<

Translate the mechanics of that specific CVE into realistic Datadog/Splunk SIEM logs for the compromised node.

Respond ONLY with valid JSON, no surrounding text:
{{
  "compromised_nodes": ["node1"],
  "threat_type": "data_exfiltration",
  "threat_ips": ["10.0.x.x"],
  "iam_roles": ["role-name-svc"],
  "red_herring_nodes": ["node"],
  "siem_evidence_template": "Log evidence matching the CVE mechanics. Use {{ip}} and {{role}}.",
  "difficulty_description": "One sentence describing the CVE and attack vector"
}}"""

    scenario = call_llm_json(prompt, 
                         temperature=0.7, 
                         model="llama-3.1-8b-instant",
                         )
    scenario["cve_context"] = live_cve_context
    return scenario


def _static_pick(difficulty: str) -> dict:
    if difficulty in ["advanced", "expert"]:
        return random.choice(_STATIC_ADVANCED)
    return random.choice(_STATIC_WARMUP)


def _normalize(raw: dict, difficulty: str = "unknown") -> dict:
    raw_nodes = raw.get("compromised_nodes", ["hr_db"])
    valid_nodes = [n for n in raw_nodes if n in INTERNAL_NODES]
    if not valid_nodes:
        valid_nodes = ["hr_db"]
    
    raw_ips = raw.get("threat_ips", [])
    while len(raw_ips) < len(valid_nodes):
        raw_ips.append(f"10.0.{random.randint(1, 10)}.{random.randint(10, 99)}")
    
    raw_roles = raw.get("iam_roles", [])
    role_defaults = ["frontend-webapp-svc", "payment-processor-svc", "hr-reader-svc"]
    while len(raw_roles) < len(valid_nodes):
        raw_roles.append(role_defaults[len(raw_roles) % len(role_defaults)])
    
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
        "difficulty_description": raw.get("difficulty_description", f"{difficulty} security incident"),
        "cve_context": raw.get("cve_context","{difficulty} security incident")
    }