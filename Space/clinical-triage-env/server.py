import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from models import (
    Action, Observation, StepResponse, ResetResponse, 
    EnvironmentState, TaskDescription, ResetRequest,
    GraderRequest, GraderResponse, BaselineRequest, BaselineResponse
)
from environment import (
    create_initial_state, get_task_description, build_observation,
    apply_action
)
from grader import grade

app = FastAPI(
    title="ClinicalTriageEnv-v1",
    description="OpenEnv compliant clinical triage simulation.",
    version="1.1.0",
)

# In-memory session store
_state: Optional[EnvironmentState] = None

# Static files for the frontend
if os.path.exists("static"):
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
def reset(request: ResetRequest):
    global _state
    try:
        _state = create_initial_state(request.task_id, seed=request.seed)
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
def state_endpoint():
    global _state
    if _state is None:
        raise HTTPException(status_code=404, detail="No active episode.")
    return _state


# ── /tasks ────────────────────────────────────────────────────────────────────

@app.get("/tasks", response_model=list[TaskDescription])
def list_tasks():
    ids = ["triage-basics", "standard-shift", "crisis-surge"]
    return [get_task_description(tid) for tid in ids]


# ── /grader ───────────────────────────────────────────────────────────────────

@app.post("/grader", response_model=GraderResponse)
def grader_endpoint(request: GraderRequest):
    try:
        result = grade(request.model_dump())
        # The grader doesn't update the global _state usually, 
        # it just provides a score for a Move.
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── /baseline ─────────────────────────────────────────────────────────────────

@app.post("/baseline", response_model=BaselineResponse)
def baseline_endpoint(request: BaselineRequest):
    """
    Calls the HF inference router with the current observation
    and returns a structured Action.
    """
    import requests

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        # Fallback to local rule-based if token missing
        from baseline import rule_based_policy
        action = rule_based_policy(request.observation)
        return BaselineResponse(
            action=action, 
            reasoning="HF_TOKEN not set. Using rule-based fallback."
        )

    obs = request.observation
    
    system_prompt = (
        "You are a clinical triage AI. Given the ED state, choose the single best action. "
        "Respond ONLY with a valid JSON object matching this schema:\n"
        f"{json.dumps(Action.model_json_schema(), indent=2)}\n"
        "Rules: Always observe_vitals before assign_triage. Return ONLY JSON."
    )

    user_prompt = f"ED State: {obs.model_dump_json()}\n\nWhat is your next action?"

    api_url = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>",
        "parameters": {"max_new_tokens": 256, "temperature": 0.1},
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        generated = resp.json()[0]["generated_text"].strip()
        
        start = generated.find("{")
        end = generated.rfind("}") + 1
        action_dict = json.loads(generated[start:end])
        return BaselineResponse(action=Action(**action_dict), reasoning="LLM choice.")
    except Exception as e:
        from baseline import rule_based_policy
        action = rule_based_policy(obs)
        return BaselineResponse(action=action, reasoning=f"Fallback due to: {e}")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.1.0"}
