"""
models.py — Pydantic data models for the Zero Trust SRE Gym.

These are the contracts between the environment, the API, and the training loop.
Clean separation between what the agent sees (Observation), what it does (Action),
and what it gets back (Reward).
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class Alert(BaseModel):
    """A security alert from the enterprise monitoring system."""
    alert_id: str
    severity: str    # FATAL, CRITICAL, WARNING, INFO
    target_node: str
    symptom: str


class Observation(BaseModel):
    """Everything the agent can see at each step."""
    
    # The current security alerts — mix of real threats and red herrings
    active_alerts: List[Alert] = Field(default_factory=list)
    
    # Output of the last tool execution (SIEM logs, ticket status, etc.)
    command_output: str = "Environment initialized. Awaiting first command."
    
    # Current production health — isolating wrong nodes drops this
    global_uptime: float = 100.0
    
    # ITIL change ticket tracking
    active_ticket_id: Optional[str] = None
    ticket_approved: bool = False
    
    # Curriculum metadata — agent can use these as hints
    difficulty: str = "warmup"
    episode_number: int = 0
    judge_persona: str = "senior"   # junior, senior, or principal


class Action(BaseModel):
    """What the agent sends to the environment each step."""
    
    # Which tool to use
    tool_name: str  # query_siem_logs | file_ticket | check_approval | isolate_node
    
    # Tool arguments — varies by tool
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    # Agent's stated reasoning — evaluated by the LLM judge on file_ticket
    justification: str = ""


class Reward(BaseModel):
    """What the environment sends back after each action."""
    value: float
    message: str


class TaskRequest(BaseModel):
    """Request body for /reset endpoint."""
    task_id: Optional[str] = "auto"
    # "auto" → use curriculum controller to pick difficulty
    # explicit id → override (for evaluation/testing)