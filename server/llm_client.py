"""
llm_client.py — Groq API wrapper for the Zero Trust SRE Gym.

Why Groq specifically: it's free-tier friendly, fast enough for real-time
episode stepping (sub-second latency on llama-3.1-8b-instant), and the
OpenAI-compatible interface makes it trivial to swap models.

All LLM calls go through here so we have one place to swap providers,
handle fallbacks, and log costs if needed.
"""

import os
import re
import json
from groq import Groq

_client = None


def get_client() -> Groq:
    """Lazy singleton — only initializes when first called."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at console.groq.com and export it before running."
            )
        _client = Groq(api_key=api_key)
    return _client


def call_llm(
    prompt: str,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.3,
    max_tokens: int = 512,
    system: str = None
) -> str:
    """
    Raw LLM call. Returns the text response string.
    
    Model choices and tradeoffs:
    - llama-3.1-8b-instant: fastest, free tier, good for judge + designer
    - llama-3.3-70b-versatile: higher quality, slower, use for complex scenarios
    - gemma2-9b-it: good middle ground, very fast
    """
    client = get_client()
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content.strip()


def call_llm_json(
    prompt: str,
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.2,
    fallback: dict = None
) -> dict:
    """
    Calls LLM and returns a parsed dict. Handles the usual LLM JSON messiness:
    - Markdown code fences (```json ... ```)  
    - Leading/trailing explanation text
    - Partial JSON with truncation
    
    If parsing fails completely, returns fallback dict if provided, else raises.
    """
    raw = call_llm(prompt, model=model, temperature=temperature)
    
    # Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?', '', raw).strip().rstrip('`').strip()
    
    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try extracting the JSON object from within surrounding text
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Nothing worked
    if fallback is not None:
        return fallback
    
    raise ValueError(f"Could not parse LLM response as JSON.\nRaw response: {raw[:300]}")