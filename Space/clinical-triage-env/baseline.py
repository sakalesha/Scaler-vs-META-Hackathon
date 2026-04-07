"""
baseline.py — Mandatory inference script for Meta PyTorch OpenEnv Hackathon
Demonstrates that ClinicalTriageEnv-v0 is learnable by running a full episode
using the HF inference router (Mistral-7B-Instruct).

Usage:
  export HF_TOKEN=hf_your_token_here
  python baseline.py

  # Or point at a deployed Space:
  python baseline.py --base-url https://your-space.hf.space
"""

import os
import sys
import json
import argparse
import requests

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:7860"
TASK_ID          = "clinical-triage-v0"


def run_episode(base_url: str, seed: int = 42, verbose: bool = True):
    """
    Runs one full episode:
      1. Starts the task (POST /tasks/{id}/start)
      2. Loops: calls /baseline to get action, /grader to score + advance state
      3. Prints step-by-step log and final cumulative score
    """
    print(f"\n{'='*60}")
    print(f"  ClinicalTriageEnv-v0 — Baseline Inference Demo")
    print(f"  Server : {base_url}")
    print(f"  Seed   : {seed}")
    print(f"{'='*60}\n")

    # ── 1. Start episode ──────────────────────────────────────────────────────
    start_resp = requests.post(
        f"{base_url}/tasks/{TASK_ID}/start",
        params={"seed": seed},
        timeout=10,
    )
    start_resp.raise_for_status()
    obs_dict = start_resp.json()

    print(f"Episode started — {len(obs_dict['patients'])} patients in queue\n")

    cumulative_score = 0.0
    step = 0
    done = False

    while not done:
        step += 1

        # ── 2. Get action from baseline (HF LLM or fallback) ─────────────────
        baseline_resp = requests.post(
            f"{base_url}/baseline",
            json={"task_id": TASK_ID, "observation": obs_dict},
            timeout=40,
        )
        baseline_resp.raise_for_status()
        baseline_data = baseline_resp.json()

        action  = baseline_data["action"]
        reasoning = baseline_data.get("reasoning", "")

        if verbose:
            print(f"Step {step:02d} | Action: {action['action_type']} "
                  f"on patient {action['patient_id']}"
                  + (f" (ESI-{action['esi_level']})" if action.get('esi_level') else ""))
            if reasoning:
                short_reason = reasoning[:80].replace("\n", " ")
                print(f"         Reasoning: {short_reason}…")

        # ── 3. Grade the action ───────────────────────────────────────────────
        # Rebuild state from current obs for grader
        grader_payload = {
            "task_id": TASK_ID,
            "action": action,
            "state": _obs_to_state(obs_dict),
        }

        grader_resp = requests.post(
            f"{base_url}/grader",
            json=grader_payload,
            timeout=15,
        )
        grader_resp.raise_for_status()
        grader_data = grader_resp.json()

        score    = grader_data["score"]
        rationale = grader_data["rationale"]
        done     = grader_data["done"]

        cumulative_score += score

        if verbose:
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"         Score: {score:.4f} [{bar}]")
            print(f"         {rationale[:100]}")
            print()

        # Update obs for next step
        obs_dict = build_obs_from_grader(grader_data)

        if done:
            break

    avg_score = cumulative_score / max(step, 1)
    print(f"{'='*60}")
    print(f"  Episode complete after {step} steps")
    print(f"  Cumulative score : {cumulative_score:.4f}")
    print(f"  Average score    : {avg_score:.4f}")
    print(f"{'='*60}\n")

    return {"steps": step, "cumulative": cumulative_score, "average": avg_score}


# ── Helper: convert obs dict back to a minimal state dict for grader ──────────

def _obs_to_state(obs: dict) -> dict:
    """Converts observation dict into the state shape the grader expects."""
    return {
        "task_id":                  obs["task_id"],
        "sim_clock_minutes":        obs["sim_clock_minutes"],
        "max_shift_minutes":        360.0,
        "action_tokens_remaining":  obs["action_tokens_remaining"],
        "beds_available":           obs["beds_available"],
        "lab_slots_available":      obs["lab_slots_available"],
        "imaging_slots_available":  obs["imaging_slots_available"],
        "patients":                 obs["patients"],
        "step_count":               obs["step_count"],
        "episode_done":             obs["episode_done"],
    }


def build_obs_from_grader(grader_data: dict) -> dict:
    """Extracts next observation from grader response."""
    ns = grader_data["next_state"]
    return {
        "task_id":                  ns["task_id"],
        "sim_clock_minutes":        ns["sim_clock_minutes"],
        "action_tokens_remaining":  ns["action_tokens_remaining"],
        "beds_available":           ns["beds_available"],
        "lab_slots_available":      ns["lab_slots_available"],
        "imaging_slots_available":  ns["imaging_slots_available"],
        "step_count":               ns["step_count"],
        "episode_done":             ns["episode_done"],
        "patients":                 ns["patients"],
        "last_action_feedback":     grader_data["rationale"],
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClinicalTriageEnv-v0 baseline runner")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="Base URL of the FastAPI server")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Episode random seed")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--quiet",    action="store_true",
                        help="Suppress per-step output")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("⚠️  HF_TOKEN not set — server will use rule-based fallback for /baseline")
        print("   Set it with: export HF_TOKEN=hf_your_token_here\n")

    all_results = []
    for ep in range(args.episodes):
        print(f"\n──── Episode {ep + 1}/{args.episodes} ────")
        result = run_episode(
            base_url=args.base_url,
            seed=args.seed + ep,
            verbose=not args.quiet,
        )
        all_results.append(result)

    if args.episodes > 1:
        avg_cumulative = sum(r["cumulative"] for r in all_results) / len(all_results)
        avg_steps      = sum(r["steps"]      for r in all_results) / len(all_results)
        print(f"\n{'='*60}")
        print(f"  Multi-episode summary ({args.episodes} episodes)")
        print(f"  Mean cumulative score : {avg_cumulative:.4f}")
        print(f"  Mean episode length   : {avg_steps:.1f} steps")
        print(f"{'='*60}")
