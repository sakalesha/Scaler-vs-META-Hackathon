from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict
import uuid

from ..models import (
    BugFixAction, BugFixObservation,
    GraderResult, BaselineResult
)
from ..environment import BugFixerEnvironment, TASKS
from ..grader import grade
from baseline import run_baseline

app = FastAPI(title="Python Bug Fixer Environment")

# In-memory session store: session_id → environment instance
sessions: Dict[str, BugFixerEnvironment] = {}


# ── Health ──────────────────────────────────────────────

@app.get("/health")
def health():
    """
    Simple health check endpoint to verify the server is running.
    """
    return {"status": "ok"}


# ── Standard OpenEnv Endpoints ───────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"

@app.post("/reset")
def reset(req: ResetRequest):
    """
    Reset the environment for a specific task and start a new session.
    Returns a unique session_id and the initial observation (state) of the bug.
    """
    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {req.task_id}")
    
    # Generate a unique ID for this episode/session
    session_id = str(uuid.uuid4())
    
    # Initialize a fresh environment instance
    env = BugFixerEnvironment(task_id=req.task_id)
    sessions[session_id] = env
    
    # Get the initial observation (buggy code, etc.)
    obs = env.reset()
    return {"session_id": session_id, "observation": obs.dict()}


@app.post("/step")
def step(session_id: str, action: BugFixAction):
    """
    Submit an action (fixed code) to the environment for grading.
    The environment will evaluate the code and return the next observation (including reward and feedback).
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
        
    env = sessions[session_id]
    obs = env.step(action)
    return {"observation": obs.dict()}


@app.get("/state")
def state(session_id: str):
    """
    Get the current internal state of a session without taking a step.
    Useful for debugging or checking progress.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return sessions[session_id].state()


# ── Required Hackathon Endpoints ─────────────────────────

@app.get("/tasks")
def get_tasks():
    """
    Retrieve a list of all available tasks in the environment.
    This allows external agents to discover what bugs are available to fix.
    """
    task_list = []
    for task_id, task in TASKS.items():
        # Dynamically construct the task schema exposing necessary fields
        task_list.append({
            "id": task_id,
            "name": task["name"],
            "description": task["description"],
            "difficulty": task["difficulty"],
            "action_schema": {
                "fields": [
                    {"name": "fixed_code", "type": "str",
                     "description": "The corrected Python code"},
                    {"name": "explanation", "type": "str",
                     "description": "What bug was fixed and why"},
                ]
            }
        })
    return {"tasks": task_list}


class GraderRequest(BaseModel):
    session_id: str
    task_id: str

@app.post("/grader")
def grader(req: GraderRequest):
    """
    Get the detailed grading results for the last step in a session.
    Provides the final score, feedback string, and breakdown of test cases passed.
    """
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
        
    env = sessions[req.session_id]
    score = env.last_score
    breakdown = env.last_breakdown
    feedback = env.last_feedback
    
    return GraderResult(
        score=score,
        breakdown=breakdown,
        feedback=feedback
    ).dict()


@app.post("/baseline")
def baseline():
    """
    Execute a baseline model evaluation (e.g., assessing an AI agent against the tasks).
    Returns the baseline metrics (scores vs expectations).
    """
    results = run_baseline()
    return results