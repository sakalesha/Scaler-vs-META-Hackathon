---
title: ClinicalTriageEnv v1
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: apache-2.0
app_port: 7860
tags:
- openenv
---

# ClinicalTriageEnv-v1

> A real-world sequential clinical triage environment for the OpenEnv Hackathon.

## What it is

An RL environment where an AI agent acts as a **triage reasoning engine** in a simulated emergency department. The agent must:

- Gather clinical information (observe vitals, order labs/imaging)
- Assign the correct ESI triage level (1 = life-threatening → 5 = non-urgent)
- Allocate limited resources (beds, lab slots, imaging)
- Disposition patients (discharge, admit, escalate)

## OpenEnv Spec Compliance

This environment implements the full **OpenEnv** interface:
- **`POST /reset`**: Initialize episode with a `task_id`.
- **`POST /step`**: Process standard `Action` and return `(Observation, Reward, Done, Info)`.
- **`GET /state`**: Return the current internal environment state.

## Tasks & Difficulty

| Task ID | Difficulty | Description |
|---------|------------|-------------|
| `triage-basics` | Easy | 3 patients, simple complaints, focus on vitals. |
| `standard-shift` | Medium | 8+ patients, stochastic arrivals, 6-hour shift. |
| `crisis-surge` | Hard | 15 patients, rapid deterioration, resource scarcity. |

## Quick start (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export HF_TOKEN=hf_your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# 3. Launch the server
uvicorn server:app --port 7860

# 4. Run the baseline evaluation (in another terminal)
python inference.py
```

## Reward Function

The environment provides a meaningful reward signal in **[0.0, 1.0]** for every step:
- **Accuracy**: Rewards correct ESI assignment based on true patient acuity.
- **Safety**: Penalizes dangerous discharges or ignoring critical (ESI-1) patients.
- **Efficiency**: Penalizes redundant vitals or wasting labs on minor cases.
- **Progress**: Rewards completing dispositions correctly.

## Environment Specs

- **Action Space**: typed `Action` (8 types, 10 patients).
- **Observation Space**: typed `Observation` (Partially observable clinical states).
- **Episode length**: Up to 50 steps (6-8 sim-hours).
- **Stochastic Arrivals**: New patients arrive based on task difficulty.
- **Clinical Deterioration**: Un-triaged patients may deteriorate over time.
