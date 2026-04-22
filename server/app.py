"""
app.py — FastAPI server for the Zero Trust SRE Gym.

Serves the OpenEnv HTTP API on port 7860 (HuggingFace Spaces default).

Endpoints:
  GET  /           → health check + curriculum summary
  POST /reset      → start new episode, get initial observation
  GET  /state      → current observation + node topology (for dashboard)
  POST /step       → execute action, get observation + reward
  GET  /history    → session action log (for dashboard reward curve)
  GET  /curriculum → detailed curriculum state (for training monitoring)
"""

import json
from fastapi import FastAPI, HTTPException
from .models import Action, Observation, TaskRequest
from .environment import ZeroTrustEnv

app = FastAPI(
    title="Zero Trust SRE Gym",
    description=(
        "RL environment where an agent learns to diagnose and contain enterprise "
        "security incidents through strict Zero Trust compliance workflows. "
        "Features adversarial scenario generation, a 3-persona LLM compliance judge, "
        "and an automatic curriculum controller."
    ),
    version="2.0.0"
)

# Single environment instance — shared across requests within one server session
env = ZeroTrustEnv()

# Session action log — written on every step, read by dashboard /history endpoint
session_history: list[dict] = []


@app.get("/")
def health_check():
    """Health check and curriculum overview."""
    return {
        "status": "ok",
        "environment": "zero-trust-sre-gym",
        "version": "2.0.0",
        "curriculum": env.curriculum.get_summary()
    }


@app.post("/reset", response_model=Observation)
async def reset_environment(request: TaskRequest):
    """
    Start a new episode. Returns the initial observation with active alerts.
    
    The adversarial designer generates a fresh scenario at every reset,
    so no two episodes are identical.
    """
    global session_history
    session_history.clear()
    
    try:
        obs = env.reset()
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.get("/state")
def get_current_state():
    """
    Current observation PLUS internal topology and curriculum state.
    Used by the Streamlit dashboard to render the service mesh and reward curve.
    
    Returns more than /reset because the dashboard needs node statuses
    which aren't part of the official agent observation (they would leak
    the compromised node to the agent).
    """
    if not env.state:
        env.reset()
    
    state_dict = env.state.model_dump()
    state_dict["nodes"] = env.nodes
    state_dict["curriculum"] = env.curriculum.get_summary()
    state_dict["episode_reward"] = env.episode_reward
    return state_dict


@app.post("/step")
def take_step(action: Action):
    """
    Execute one agent action. Core training endpoint.
    
    Returns:
      observation: updated environment state
      reward: numeric value + explanation message
      done: whether episode has ended
      info: difficulty, episode_reward, judge_persona, steps_remaining
    """
    global session_history
    
    if not env.state:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first."
        )
    
    obs, reward, done, info = env.step(action)
    
    # Append to session history for dashboard
    session_history.append({
        "step": env.steps,
        "action": action.tool_name,
        "target": json.dumps(action.payload)[:80],
        "reward": reward.value,
        "message": reward.message,
        "cumulative_reward": env.episode_reward
    })
    
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }


@app.get("/history")
def get_history():
    """
    Full session action log for the current episode.
    Used by dashboard to render the reward curve and audit trail.
    """
    return {
        "history": session_history,
        "curriculum": env.curriculum.get_summary(),
        "episode_reward": env.episode_reward,
        "episode_done": env.done
    }


@app.get("/curriculum")
def get_curriculum():
    """
    Detailed curriculum state for training monitoring.
    Shows mastery per threat type, difficulty, recent rewards.
    """
    return env.curriculum.get_summary()


@app.post("/curriculum/reset")
def reset_curriculum():
    """Hard reset the curriculum — for demos and evaluation runs."""
    env.curriculum.reset_mastery()
    return {"status": "ok", "message": "Curriculum reset to episode 0"}