from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class Alert(BaseModel):
    alert_id: str
    severity: str
    target_node: str
    symptom: str

class Observation(BaseModel):
    active_alerts: List[Alert] = Field(default_factory=list)
    command_output: str = "Awaiting command..."
    global_uptime: float = 100.0
    active_ticket_id: Optional[str] = None
    ticket_approved: bool = False

class Action(BaseModel):
    tool_name: str 
    payload: Dict[str, Any]
    justification: str 

class Reward(BaseModel):
    value: float
    message: str

class TaskRequest(BaseModel):
    task_id: Optional[str] = "level_3_insider_threat"