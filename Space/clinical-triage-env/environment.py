# What is the purpose of this file in this project?
# This file defines the environment for the clinical triage environment.
# It includes:
# - The environment state
# - The action space
# - The observation space
# - The reward function
# - The reset function
# - The step function
# - The render function
# - The close function  

import random
import uuid
import json
from typing import Tuple, List, Dict, Any, Optional
from models import (
    PatientState, EnvironmentState, Action,
    Observation, TaskDescription, StepResponse
)

# ── Synthetic patient data pools ──────────────────────────────────────────────

COMPLAINTS = {
    1: ["chest pain with diaphoresis", "unresponsive", "severe respiratory distress",
        "active seizure", "major trauma"],
    2: ["altered mental status", "severe abdominal pain", "stroke symptoms",
        "high fever with stiff neck", "sudden vision loss"],
    3: ["moderate abdominal pain", "back pain with weakness", "vomiting x3",
        "fracture suspected", "asthma exacerbation"],
    4: ["minor laceration", "ear pain", "urinary symptoms", "mild fever",
        "sprained ankle"],
    5: ["prescription refill", "rash 3 days old", "cold symptoms",
        "routine follow-up", "mild headache"],
}

VITALS_RANGES = {
    1: dict(hr=(110, 160), sbp=(60, 89),  spo2=(0.80, 0.89), rr=(25, 35), temp=(38.5, 40.5)),
    2: dict(hr=(100, 130), sbp=(85, 100), spo2=(0.88, 0.93), rr=(20, 28), temp=(38.0, 39.5)),
    3: dict(hr=(88,  110), sbp=(100,125), spo2=(0.93, 0.97), rr=(16, 22), temp=(37.0, 38.5)),
    4: dict(hr=(70,  95),  sbp=(115,135), spo2=(0.96, 0.99), rr=(14, 18), temp=(36.5, 37.5)),
    5: dict(hr=(60,  85),  sbp=(115,130), spo2=(0.97, 1.00), rr=(12, 16), temp=(36.0, 37.2)),
}

LAB_RESULTS = {
    1: ["Troponin elevated 12x. Lactate 4.2. WBC 18k.",
        "CBC: Hgb 6.1. BMP: K 6.8. Glucose 42."],
    2: ["Troponin 0.08 (borderline). BNP elevated.",
        "WBC 16k with left shift. Lipase 850."],
    3: ["BMP normal. CBC: Hgb 10.2. UA: positive nitrites.",
        "Lipase mildly elevated 180. LFTs pending."],
    4: ["CBC and BMP within normal limits.", "UA trace blood only."],
    5: ["All labs within normal limits.", "No significant findings."],
}

IMAGING_RESULTS = {
    1: ["CXR: bilateral infiltrates, tension pneumothorax suspected.",
        "CT head: large hemorrhagic stroke confirmed."],
    2: ["CXR: cardiomegaly, pulmonary edema.",
        "CT abdomen: appendicitis with perforation risk."],
    3: ["CXR: patchy opacity, possible pneumonia.",
        "X-ray: non-displaced fracture confirmed."],
    4: ["CXR: clear.", "X-ray: soft tissue swelling, no fracture."],
    5: ["No acute findings.", "Normal study."],
}

# ── Environment Support ──────────────────────────────────────────────────────

def _sample_patient(esi: int) -> PatientState:
    return PatientState(
        patient_id=str(uuid.uuid4())[:8],
        wait_minutes=round(random.uniform(2, 45), 1),
        chief_complaint=random.choice(COMPLAINTS[esi]),
        age_group=random.choice(["infant", "child", "adult", "elderly"]),
        gender=random.choice(["M", "F", "Other"]),
        true_esi=esi,
    )

# This function is used to sample vitals for a patient based on their ESI level.
# It is used to generate realistic vital signs for the patients in the environment.         
def _sample_vitals(patient: PatientState) -> PatientState:
    esi = patient.true_esi or 3
    v = VITALS_RANGES[esi]
    patient.heart_rate        = random.randint(*v["hr"])
    patient.systolic_bp       = random.randint(*v["sbp"])
    patient.spo2              = round(random.uniform(*v["spo2"]), 2)
    patient.respiratory_rate  = random.randint(*v["rr"])
    patient.temperature       = round(random.uniform(*v["temp"]), 1)
    patient.has_deteriorated  = False
    return patient

# ── Core API ──────────────────────────────────────────────────────────────────

# It is used to generate the initial state of the environment.
def create_initial_state(task_id: str, seed: Optional[int] = None) -> EnvironmentState:
    if seed is not None:
        random.seed(seed)
    
    if task_id == "triage-basics":
        # Easy: 3 patients, no complicated constraints
        patients = [_sample_patient(random.choice([2, 3, 4])) for _ in range(3)]
        return EnvironmentState(
            task_id=task_id,
            patients=patients,
            action_tokens_remaining=10,
            beds_available=5,
            max_shift_minutes=120
        )
    elif task_id == "standard-shift":
        # Medium: 8 patients, stochastic arrivals
        esi_dist = [1, 2, 2, 3, 3, 3, 3, 4, 4, 5]
        patients = [_sample_patient(random.choice(esi_dist)) for _ in range(5)]
        return EnvironmentState(
            task_id=task_id,
            patients=patients,
            action_tokens_remaining=25,
            beds_available=10,
            max_shift_minutes=360
        )
    elif task_id == "crisis-surge":
        # Hard: Many patients, rapid deterioration
        esi_dist = [1, 1, 2, 2, 2, 3, 4]
        patients = [_sample_patient(random.choice(esi_dist)) for _ in range(8)]
        return EnvironmentState(
            task_id=task_id,
            patients=patients,
            action_tokens_remaining=30,
            beds_available=4, # Resource scarcity
            max_shift_minutes=480
        )
    else:
        # Fallback
        return create_initial_state("standard-shift", seed)

# It is used to apply an action to the environment. 
# It returns the next state, reward, done, and info.  
def apply_action(state: EnvironmentState, action: Action) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
    """
    Core transition function. Returns (next_state, reward, done, info).
    Integrates grading logic for the reward signal.
    """
    patient = next((p for p in state.patients if p.patient_id == action.patient_id), None)
    
    # Baseline checks
    if patient is None or patient.is_dispositioned:
        reward = 0.05
        feedback = "Invalid target: patient not found or already dispositioned."
    else:
        reward, feedback = _calculate_reward(state, action, patient)
        
        # Apply the physical action
        state, action_feedback = _execute_action_logic(state, action, patient)
        feedback += f" | {action_feedback}"

    # Deterioration (higher probability in hard mode)
    det_rate = 0.05 if state.task_id == "crisis-surge" else 0.02
    det_msg = _process_deterioration(state, det_rate)
    if det_msg:
        feedback += f" | {det_msg}"

    # Advance clock
    state.sim_clock_minutes += 5
    state.step_count += 1

    # Arrivals (more frequent in standard/surge)
    if state.task_id != "triage-basics":
        arrival_freq = 4 if state.task_id == "crisis-surge" else 6
        if state.step_count % arrival_freq == 0 and len(state.patients) < 15:
            new_pt = _sample_patient(random.choice([1, 2, 3, 4, 5]))
            state.patients.append(new_pt)
            feedback += f" | 🚑 New arrival: {new_pt.patient_id}"

    # Terminal conditions
    done = (
        state.sim_clock_minutes >= state.max_shift_minutes
        or all(p.is_dispositioned for p in state.patients)
        or state.step_count >= 50
    )
    state.episode_done = done
    
    info = {"feedback": feedback}
    return state, reward, done, info

# It is used to execute the action logic.
def _execute_action_logic(state: EnvironmentState, action: Action, patient: PatientState) -> Tuple[EnvironmentState, str]:
    atype = action.action_type
    feedback = ""

    if atype == "observe_vitals":
        if state.action_tokens_remaining >= 1:
            _sample_vitals(patient)
            state.action_tokens_remaining -= 1
            feedback = f"Vitals: HR:{patient.heart_rate} BP:{patient.systolic_bp} SpO2:{patient.spo2}"
        else:
            feedback = "No tokens left."
    
    elif atype == "order_labs":
        if state.action_tokens_remaining >= 2 and state.lab_slots_available >= 1:
            patient.labs_result = random.choice(LAB_RESULTS.get(patient.true_esi or 3, LAB_RESULTS[3]))
            state.lab_slots_available -= 1
            state.action_tokens_remaining -= 2
            feedback = f"Labs: {patient.labs_result}"
        else:
            feedback = "Insufficient resources for labs."
            
    elif atype == "order_imaging":
        if state.action_tokens_remaining >= 3 and state.imaging_slots_available >= 1:
            patient.imaging_result = random.choice(IMAGING_RESULTS.get(patient.true_esi or 3, IMAGING_RESULTS[3]))
            state.imaging_slots_available -= 1
            state.action_tokens_remaining -= 3
            feedback = f"Imaging: {patient.imaging_result}"
        else:
            feedback = "Insufficient resources for imaging."

    elif atype == "assign_triage":
        patient.assigned_esi = action.esi_level
        feedback = f"Assigned ESI-{action.esi_level}"

    elif atype == "allocate_bed":
        if state.beds_available > 0:
            state.beds_available -= 1
            feedback = "Bed allocated."
        else:
            feedback = "No beds available."

    elif atype in ("discharge", "admit"):
        patient.is_dispositioned = True
        state.beds_available += 1
        feedback = f"Patient {atype}ed."

    elif atype == "escalate":
        patient.assigned_esi = 1
        patient.is_dispositioned = True
        feedback = "ESCALATED."

    return state, feedback

# It is used to calculate the reward for an action. 
def _calculate_reward(state: EnvironmentState, action: Action, patient: PatientState) -> Tuple[float, str]:
    """Integrated Reward Shaping logic."""
    score = 0.1
    rationale = ""
    atype = action.action_type
    true_esi = patient.true_esi or 3
    has_info = patient.heart_rate is not None

    if atype == "assign_triage":
        if action.esi_level is None:
            score = 0.05
            rationale = "Missing ESI level."
        else:
            diff = abs(action.esi_level - true_esi)
            base = {0: 1.0, 1: 0.65, 2: 0.35}.get(diff, 0.05)
            bonus = 0.10 if has_info else -0.10
            score = min(1.0, max(0.05, base + bonus))
            rationale = f"Triage accuracy: {base:.2f}"

    elif atype == "observe_vitals":
        score = 0.90 if true_esi <= 2 else 0.50
        rationale = "Clinical assessment."

    elif atype == "escalate":
        score = 1.0 if true_esi == 1 else 0.20
        rationale = "Escalation logic."

    elif atype in ("discharge", "admit"):
        if patient.assigned_esi is None:
            score = 0.20
            rationale = "Untriaged disposition."
        elif atype == "discharge" and patient.assigned_esi >= 4:
            score = 0.90
            rationale = "Correct discharge."
        elif atype == "admit" and patient.assigned_esi <= 2:
            score = 0.90
            rationale = "Correct admission."
        else:
            score = 0.40
            rationale = "Suboptimal disposition."

    # Critical delay penalty
    urgents = [p for p in state.patients if p.true_esi == 1 and not p.is_dispositioned and p.assigned_esi is None]
    if urgents and atype not in ("escalate", "observe_vitals", "assign_triage"):
        score = max(0.0, score - 0.2)
        rationale += " [Critical patient ignored]"

    return round(score, 3), rationale

# It is used to process the deterioration of patients.
# deterioration: It is used to process the deterioration of patients.     
def _process_deterioration(state: EnvironmentState, rate: float) -> str:
    alerts = []
    for p in state.patients:
        if p.is_dispositioned or p.true_esi == 1:
            continue
        if random.random() < rate:
            p.true_esi -= 1
            p.heart_rate = None # Reset vitals for "re-discovery"
            p.has_deteriorated = True
            alerts.append(f"Patient {p.patient_id} crashed!")
    return " ".join(alerts)

# It is used to build the observation for the agent.    
def build_observation(state: EnvironmentState, feedback: str = "") -> Observation:
    return Observation(
        task_id=state.task_id,
        sim_clock_minutes=state.sim_clock_minutes,
        action_tokens_remaining=state.action_tokens_remaining,
        beds_available=state.beds_available,
        lab_slots_available=state.lab_slots_available,
        imaging_slots_available=state.imaging_slots_available,
        step_count=state.step_count,
        episode_done=state.episode_done,
        patients=state.patients,
        last_action_feedback=feedback,
    )

# It is used to get the task description based on the task_id.
def get_task_description(task_id: str) -> TaskDescription:
    # This should return the description based on the task_id
    tasks = {
        "triage-basics": "Easy: Triage 3 patients. Minimal constraints.",
        "standard-shift": "Medium: 6-hour shift. Stochastic arrivals. Resource management.",
        "crisis-surge": "Hard: High pressure. Rapid deterioration. Resource scarcity."
    }
    return TaskDescription(
        task_id=task_id,
        description=tasks.get(task_id, "Standard clinical triage task."),
        action_schema=Action.model_json_schema(),
        observation_schema=Observation.model_json_schema()
    )
