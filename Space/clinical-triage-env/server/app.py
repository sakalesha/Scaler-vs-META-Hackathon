"""
ClinicalTriageEnv-v1 — OpenEnv Compliant FastAPI Server
Endpoints:
  POST /reset  → Initialize a new health triage episode
  POST /step   → Take an action, get (obs, reward, done, info)
  GET  /state  → Get the full internal state
"""

import sys
import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure the root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models import (
    Action, Observation, StepResponse, ResetResponse, 
    EnvironmentState, TaskDescription, ResetRequest
)
from environment import (
    create_initial_state, get_task_description, build_observation,
    apply_action
)

app = FastAPI(
    title="ClinicalTriageEnv-v1",
    description="OpenEnv compliant clinical triage simulation.",
    version="1.1.0",
)

# In-memory session store
_state: Optional[EnvironmentState] = None

STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Serves the interactive dashboard."""
    try:
        with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>ClinicalTriageEnv-v1</h1><p>Static files not found.</p>", status_code=404)

# ── OpenEnv Standard Endpoints ───────────────────────────────────────────────

@app.post("/reset", response_model=ResetResponse)
async def reset(
    request: Request,
    task_id: Optional[str] = None, 
    seed: Optional[int] = None
):
    global _state
    # Debug: Log the incoming request to help diagnose validator issues
    body = await request.body()
    print(f"DEBUG: /reset called with Query: {request.query_params}, Body: {body.decode()}")

    # Determine task_id from Query, or Body, or Default
    actual_task_id = task_id
    if not actual_task_id and body:
        try:
            data = json.loads(body)
            actual_task_id = data.get("task_id")
        except:
            pass
    
    if not actual_task_id:
        actual_task_id = "triage-basics"

    try:
        _state = create_initial_state(actual_task_id, seed=seed)
        obs = build_observation(_state, f"Episode reset ({actual_task_id}). Welcome to the ED.")
        return ResetResponse(observation=obs)
    except Exception as e:
        print(f"ERROR in /reset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    global _state
    if _state is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")
    
    try:
        next_state, reward, done, info = apply_action(_state, action)
        _state = next_state
        obs = build_observation(_state, info.get("feedback", ""))
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", response_model=EnvironmentState)
def state():
    global _state
    if _state is None:
        raise HTTPException(status_code=404, detail="No active episode.")
    return _state

# ── Discovery ─────────────────────────────────────────────────────────────────

@app.get("/tasks", response_model=list[TaskDescription])
def list_tasks():
    ids = ["triage-basics", "standard-shift", "crisis-surge"]
    return [get_task_description(tid) for tid in ids]

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.1.0"}

def main():
    """Entry point for the openenv-server script."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
