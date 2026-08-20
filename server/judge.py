"""
judge.py — The Compliance Judge for Zero Trust SRE Gym.
"""

from .llm_client import call_llm_json
import groq

PERSONAS = {
    "junior": {
        "identity": (
            "You are a Junior SRE (1-2 years experience) reviewing incident tickets. "
            "You are supportive and want to encourage good practices. "
            "You give partial credit for partially correct reasoning — if the agent tried "
            "to investigate and mentioned something relevant, lean toward approval."
        ),
        "passing_threshold": 0.2,
        "score_scale": "0.8 = cited any forensic detail, 0.5 = mentioned investigation, 0.2 = vague but plausible, -0.5 = wrong node"
    },
    "senior": {
        "identity": (
            "You are a Senior SRE (6 years experience) running incident response. "
            "You follow standard enterprise Change Management process. "
            "You expect specific forensic evidence: IP addresses, IAM role names, log patterns, or timestamps. "
            "Generic phrases like 'found suspicious activity' are NOT sufficient. "
            "You approve if evidence is specific. You reject if it's vague."
        ),
        "passing_threshold": 0.3,
        "score_scale": "1.0 = cites specific IP/role/log, 0.5 = partially specific, 0.0 = generic, -1.0 = completely unfounded"
    },
    "principal": {
        "identity": (
            "You are a Principal SRE and acting CISO. You have final authority on all isolation decisions. "
            "You are extremely strict. You need: the exact threat indicator (IP or IAM role), "
            "the specific anomaly observed (bytes transferred, role assumption event, lateral query), "
            "and confirmation that the agent investigated BEFORE filing. "
            "Vague language gets an automatic -1.0. 'I found bad activity' is a rejection. "
            "You penalize agents that would have caused collateral damage."
        ),
        "passing_threshold": 0.4,
        "score_scale": "1.0 = precise forensic citation, 0.3 = mostly specific with minor gaps, -0.5 = vague, -1.0 = no evidence"
    }
}


def evaluate_ticket(
    node: str,
    justification: str,
    siem_evidence: str,
    persona: str = "senior"
) -> tuple[float, str]:
    """
    Evaluate the ticket justification. Returns (score, reason).
    If the LLM call fails, returns (0.0, "LLM unavailable – default neutral score")
    so the demo can continue without crashing.
    """
    persona_config = PERSONAS.get(persona, PERSONAS["senior"])
    
    prompt = f"""{persona_config["identity"]}

An autonomous SRE agent wants to file an isolation ticket.

Target node: {node}
Agent's justification: "{justification}"
SIEM evidence that was available to the agent: "{siem_evidence}"

Scoring scale: {persona_config["score_scale"]}

Evaluate the justification. Consider:
1. Does it cite specific forensic indicators (IPs, IAM roles, anomaly metrics)?
2. Does it match the available SIEM evidence?  
3. Does it target the correct node?
4. Would this justify isolating a production service under enterprise policy?

Respond ONLY with valid JSON (absolutely no text outside the JSON):
{{"score": <float between -1.0 and 1.0>, "reason": "<one clear sentence>", "approved": <true or false>}}"""

    try:
        result = call_llm_json(
            prompt,
            temperature=0.15,
            fallback={"score": 0.0, "reason": "Judge service unavailable — defaulting to neutral", "approved": False}
        )
    except (groq.NotFoundError, groq.BadRequestError, Exception) as e:
        # If the LLM model is not found or any other error, return a neutral score
        return 0.0, f"LLM evaluation failed ({str(e)}) – defaulting to neutral score."

    raw_score = float(result.get("score", 0.0))
    reason = result.get("reason", "No reason provided")
    score = max(-1.0, min(1.0, raw_score))
    return score, reason


def evaluate_resolution(
    action_history: list,
    compromised_node: str,
    final_state: dict,
    persona: str = "senior"
) -> tuple[float, str]:
    """
    End-of-episode workflow evaluation. Falls back to a default score if LLM fails.
    """
    persona_config = PERSONAS.get(persona, PERSONAS["senior"])
    
    if not action_history:
        return 0.0, "No actions taken — cannot evaluate workflow."
    
    summary_lines = []
    for i, action in enumerate(action_history[-10:], 1):
        tool = action.get("tool_name", "unknown")
        payload = str(action.get("payload", {}))[:60]
        result = str(action.get("result", ""))[:80]
        summary_lines.append(f"Step {i}: {tool}({payload}) → {result}")
    
    actions_text = "\n".join(summary_lines)
    
    prompt = f"""{persona_config["identity"]}

Review this complete Zero Trust incident response workflow:

Actions taken by the agent:
{actions_text}

Ground truth: compromised node was '{compromised_node}'
Final uptime: {final_state.get('uptime', 0)}%

Score the workflow quality on the 5 SRE phases:
1. TRIAGE: Did the agent read and prioritize alerts before acting?
2. INVESTIGATE: Did the agent query SIEM on the correct node?
3. DOCUMENT: Did the agent file a ticket with forensic evidence?
4. AUTHORIZE: Did the agent check approval before isolating?
5. REMEDIATE: Did the agent isolate the correct node?

Respond ONLY with valid JSON:
{{"phase_score": <float 0.0-1.0>, "phases_completed": ["TRIAGE", "INVESTIGATE", ...], "narrative": "<2 sentences on what the agent did well and what it should improve>"}}"""

    try:
        result = call_llm_json(
            prompt,
            temperature=0.2,
            fallback={"phase_score": 0.5, "phases_completed": [], "narrative": "Workflow evaluation unavailable."}
        )
    except Exception:
        return 0.5, "LLM evaluation failed – defaulting to neutral workflow score."

    phase_score = float(result.get("phase_score", 0.5))
    phase_score = max(0.0, min(1.0, phase_score))
    narrative = result.get("narrative", "")
    return phase_score, narrative