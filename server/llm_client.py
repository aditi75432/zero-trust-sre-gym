"""
llm_client.py — Groq API wrapper for the Zero Trust SRE Gym.
"""

import os
import re
import json
from groq import Groq

# Default model – can be overridden by environment variable GROQ_MODEL
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at console.groq.com and export it before running."
            )
        _client = Groq(api_key=api_key, timeout=30.0)
    return _client


def call_llm(
    prompt: str,
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
    system: str = None
) -> str:
    if model is None:
        model = DEFAULT_MODEL

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
    model: str = None,
    temperature: float = 0.2,
    fallback: dict = None
) -> dict:
    if model is None:
        model = DEFAULT_MODEL

    raw = call_llm(prompt, model=model, temperature=temperature)
    
    # Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?', '', raw).strip().rstrip('`').strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    if fallback is not None:
        return fallback
    
    raise ValueError(f"Could not parse LLM response as JSON.\nRaw response: {raw[:300]}")