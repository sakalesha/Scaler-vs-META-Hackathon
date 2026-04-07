import asyncio
import os
import json
import textwrap
import requests
from typing import List, Optional, Dict, Any
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Local environment URL (for testing)
ENV_URL = os.getenv("ENV_URL") or "http://localhost:7860"

MAX_STEPS = 10
TEMPERATURE = 0.1

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a clinical triage AI engine.
    Your goal is to manage an Emergency Department shift.
    - Observe vitals on new patients.
    - Assign accurate ESI levels (1=critical, 5=non-urgent).
    - Allocate resources (beds/labs) appropriately.
    - Disposition (discharge/admit/escalate) when ready.
    Respond ONLY with a valid JSON object representing the next Action.
    Schema: {"patient_id": "str", "action_type": "str", "esi_level": int|null, "notes": "str|null"}
    Action types: observe_vitals, order_labs, order_imaging, assign_triage, allocate_bed, discharge, admit, escalate
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ── Environment Client ────────────────────────────────────────────────────────

class TriageEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def reset(self, task_id: str, seed: int = 42):
        resp = requests.post(f"{self.base_url}/reset", params={"task_id": task_id, "seed": seed})
        resp.raise_for_status()
        return resp.json()

    async def step(self, action: Dict[str, Any]):
        resp = requests.post(f"{self.base_url}/step", json=action)
        resp.raise_for_status()
        return resp.json()

# ── Inference Logic ───────────────────────────────────────────────────────────

def get_model_action(client: OpenAI, observation: Dict[str, Any]) -> Dict[str, Any]:
    # Simplify observation for the prompt
    obs_summary = {
        "clock": observation["sim_clock_minutes"],
        "tokens": observation["action_tokens_remaining"],
        "beds": observation["beds_available"],
        "patients": [
            {
                "id": p["patient_id"],
                "comp": p["chief_complaint"],
                "esi": p["assigned_esi"],
                "vitals": "known" if p["heart_rate"] else "hidden",
                "done": p["is_dispositioned"]
            }
            for p in observation["patients"]
        ],
        "feedback": observation["last_action_feedback"]
    }
    
    user_prompt = f"Current State: {json.dumps(obs_summary)}\n\nWhat is your next action?"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"}
        )
        text = (completion.choices[0].message.content or "").strip()
        return json.loads(text)
    except Exception as exc:
        # Simple fallback
        if observation["patients"]:
            return {"patient_id": observation["patients"][0]["patient_id"], "action_type": "observe_vitals"}
        return {"patient_id": "none", "action_type": "wait"}

async def run_task(task_id: str):
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = TriageEnvClient(ENV_URL)
    
    log_start(task=task_id, env="clinical-triage", model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    success = False
    
    try:
        reset_data = await env.reset(task_id)
        obs = reset_data["observation"]
        
        for step in range(1, MAX_STEPS + 1):
            if obs["episode_done"]:
                break
                
            action_dict = get_model_action(client, obs)
            step_data = await env.step(action_dict)
            
            obs = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=json.dumps(action_dict), reward=reward, done=done, error=None)
            
            if done:
                break
        
        total_reward = sum(rewards)
        # Success if average reward is high enough
        final_score = total_reward / steps_taken if steps_taken > 0 else 0
        success = final_score > 0.5
        
    finally:
        # Calculate final normalized score in [0, 1]
        final_score = sum(rewards) / steps_taken if steps_taken > 0 else 0.0
        final_score = min(max(final_score, 0.0), 1.0)
        success = final_score >= 0.5
        
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

async def main():
    # The hackathon evaluation will call the script. 
    # Usually it might run one task or all three. 
    # Here we run the task specified in the environment or default to easy.
    task_id = os.getenv("TASK_ID", "triage-basics")
    await run_task(task_id)

if __name__ == "__main__":
    asyncio.run(main())
