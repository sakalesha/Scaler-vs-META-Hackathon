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
| `triage-basics` | Easy | Triage 3 incoming patients. Focus on vitals. |
| `standard-shift` | Medium | 8+ patients, stochastic arrivals, 6-hour shift. |
| `crisis-surge` | Hard | 15 patients, rapid deterioration, resource scarcity. |

## Quick Start (Local)

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**
   ```bash
   export HF_TOKEN=hf_your_token_here
   ```

3. **Launch the server**
   ```bash
   uvicorn server:app --port 7860
   ```

4. **Run evaluation**
   ```bash
   python inference.py
   ```

## Reward Function

The environment provides a reward signal in **[0.0, 1.0]**:
- **Accuracy**: Rewards correct ESI assignment.
- **Safety**: Penalizes dangerous discharges or ignoring critical patients.
- **Efficiency**: Penalizes redundant tests.

## Environment Specs

- **Action Space**: Typed `Action` (8 types).
- **Observation Space**: Typed `Observation` (Partially observable).
- **Episode Length**: Up to 50 steps.
- **Deterioration**: Un-triaged patients may deteriorate over time.
