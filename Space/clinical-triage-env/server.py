"""
ClinicalTriageEnv-v1 — OpenEnv Compliant FastAPI Server
Endpoints:
  POST /reset  → Initialize a new health triage episode
  POST /step   → Take an action, get (obs, reward, done, info)
  GET  /state  → Get the full internal state
"""

import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from models import (
    Action, Observation, StepResponse, ResetResponse, 
    EnvironmentState, TaskDescription
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
# In a production OpenEnv, this might handle multiple concurrent users
_state: Optional[EnvironmentState] = None


app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Serves the interactive dashboard."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>ClinicalTriageEnv-v1</h1><p>Static files not found.</p>", status_code=404)

# ── OpenEnv Standard Endpoints ───────────────────────────────────────────────

@app.post("/reset", response_model=ResetResponse)
def reset(
    task_id: str = Body(..., embed=True),
    seed: Optional[int] = Body(None, embed=True)
):
    global _state
    try:
        _state = create_initial_state(task_id, seed=seed)
        obs = build_observation(_state, "Episode reset. Welcome to the ED.")
        return ResetResponse(observation=obs)
    except Exception as e:
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
