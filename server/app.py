import json
import requests as _req
from fastapi import FastAPI, HTTPException
from .models import Action, Observation, TaskRequest
from .environment import ZeroTrustEnv

app = FastAPI(
    title="Zero Trust SRE Gym",
    description=(
        "RL environment where an agent learns to diagnose and contain enterprise "
        "security incidents through strict Zero Trust compliance workflows. "
        "Features adversarial scenario generation, a 3-persona LLM compliance judge, "
        "an automatic curriculum controller, real microservice attack simulation, "
        "and Zero Trust policy enforcement as hard MDP constraints."
    ),
    version="3.0.0",
)

env = ZeroTrustEnv()
session_history: list[dict] = []

SERVICE_PORTS = {"frontend": 5003, "payment": 5004, "hr_db": 5005}

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "environment": "zero-trust-sre-gym",
        "version": "3.0.0",
        "curriculum": env.curriculum.get_summary(),
    }

@app.post("/reset", response_model=Observation)
async def reset_environment(request: TaskRequest):
    global session_history
    session_history.clear()
    try:
        obs = env.reset()
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.get("/state")
def get_current_state():
    if not env.state:
        env.reset()
    state_dict = env.state.model_dump()
    state_dict["nodes"]          = env.nodes
    state_dict["curriculum"]     = env.curriculum.get_summary()
    state_dict["episode_reward"] = env.episode_reward
    state_dict["cve_context"]    = env.cve_context
    state_dict["siem_evidence_template"] = env.siem_evidence_template
    return state_dict

@app.post("/step")
def take_step(action: Action):
    global session_history

    if not env.state:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first.",
        )

    result = env.step(action)
    obs = result.observation
    reward = result.reward
    terminated = result.terminated
    truncated = result.truncated
    info = result.info

    session_history.append({
        "step":             env.steps,
        "action":           action.tool_name,
        "target":           json.dumps(action.payload)[:80],
        "reward":           reward.value,
        "message":          reward.message,
        "cumulative_reward": env.episode_reward,
    })

    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "terminated":  terminated,              # OpenEnv Validator
        "truncated":   truncated,               # OpenEnv Validator
        "done":        terminated or truncated, # My Notebook
        "info":        info,
    }
@app.get("/history")
def get_history():
    return {
        "history":       session_history,
        "curriculum":    env.curriculum.get_summary(),
        "episode_reward": env.episode_reward,
        "episode_done":  env.done,
    }

@app.get("/curriculum")
def get_curriculum():
    return env.curriculum.get_summary()

@app.post("/curriculum/reset")
def reset_curriculum():
    env.curriculum.reset_mastery()
    return {"status": "ok", "message": "Curriculum reset to episode 0"}

@app.get("/services")
def get_service_health():
    results = {}
    for name, port in SERVICE_PORTS.items():
        try:
            r = _req.get(f"http://localhost:{port}/health", timeout=1.5)
            data = r.json() if r.status_code == 200 else {}
            results[name] = {"reachable": True, **data}
        except Exception:
            results[name] = {"reachable": False, "status": "offline"}
    return results

@app.get("/services/{service}/metrics")
def get_service_metrics(service: str):
    port = SERVICE_PORTS.get(service)
    if not port:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service}")
    try:
        r = _req.get(f"http://localhost:{port}/metrics", timeout=2)
        return r.json()
    except Exception:
        raise HTTPException(status_code=503, detail=f"Service {service} unreachable")

@app.get("/services/{service}/logs")
def get_service_logs(service: str):
    port = SERVICE_PORTS.get(service)
    if not port:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service}")
    try:
        r = _req.get(f"http://localhost:{port}/logs", timeout=2)
        return r.json()
    except Exception:
        raise HTTPException(status_code=503, detail=f"Service {service} unreachable")

def main():
    import uvicorn
   
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()