from fastapi import FastAPI, HTTPException
from .models import Action, Observation, TaskRequest
from .environment import ZeroTrustEnv
import json

app = FastAPI(title="Zero Trust SRE OpenEnv")

# This is our lightning-fast hybrid state machine
env = ZeroTrustEnv()

# Global tracker so your Streamlit Dashboard can plot the live reward curves
session_history = []

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Zero Trust Governance Engine is active."}

@app.post("/reset", response_model=Observation)
async def reset_environment(request: TaskRequest):
    global session_history
    session_history.clear()
    try:
        return env.reset()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=Observation)
def get_current_state():
    """Endpoint for the Streamlit dashboard to poll the live state."""
    if not env.state:
        env.reset()
    return env.state

@app.get("/history")
def get_history():
    """Endpoint for the Streamlit dashboard to plot the reward graph."""
    return {
        "history": session_history,
        "task": "level_1_brute_force"
    }

@app.post("/step")
def take_step(action: Action):
    global session_history
    if not env.state:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    # Execute the action against the real Docker/Jira hybrid environment
    obs, reward, done, info = env.step(action)
    
    # Log the action for the live demo dashboard
    # We map the new schema (tool_name/payload) to the old dashboard format
    session_history.append({
        "step": env.steps,
        "action": action.tool_name,
        "target": json.dumps(action.payload), # Convert payload dict to string for the UI
        "reward": reward.value,
        "message": reward.message
    })
    
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }