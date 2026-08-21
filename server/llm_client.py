"""
llm_client.py — Groq API wrapper for the Zero Trust SRE Gym.
"""

import os
import re
import json
from groq import Groq

# Default model – can be overridden by environment variable GROQ_MODEL
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

_client = None


def get_client() -> Groq:
    """Lazy-initialise the Groq client with the API key."""
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
    max_tokens: int = 1024,   # Increased to avoid truncation
    system: str = None
) -> str:
    """
    Call the Groq LLM with a user prompt.
    """
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
    """
    Call the LLM and parse the response as JSON.
    Tries multiple extraction strategies to be robust against markdown and extra text.
    """
    if model is None:
        model = DEFAULT_MODEL

    raw = call_llm(prompt, model=model, temperature=temperature)
    
    # 1. Remove markdown code fences
    cleaned = re.sub(r'```(?:json)?', '', raw).strip().rstrip('`').strip()
    
    # 2. Try to find a JSON object with balanced braces (handles nested objects up to 2 levels)
    #    This pattern is sufficient for our simple scenario structures.
    match = re.search(r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})', cleaned, re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # fall through to other strategies
    
    # 3. If fallback is provided, return it (helps keep the system running)
    if fallback is not None:
        return fallback
    
    # 4. If all else fails, raise an error
    raise ValueError(
        f"Could not parse LLM response as JSON.\nRaw response: {raw[:300]}"
    )