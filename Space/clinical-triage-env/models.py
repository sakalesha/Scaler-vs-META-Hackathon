# This file defines the data models for the clinical triage environment.
# It includes:
# - Enumerations for ESI levels and action types
# - Patient state model
# - Environment state model
# - Action model
# - Observation model
# - Task description model
# - Grader request and response models
# - Baseline request and response models   


from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import IntEnum


# ── Enumerations ──────────────────────────────────────────────────────────────

class ESILevel(IntEnum):
    IMMEDIATE   = 1   # Life-threatening
    EMERGENT    = 2   # High risk
    URGENT      = 3   # Stable but needs care
    LESS_URGENT = 4   # Minor problem
    NON_URGENT  = 5   # Routine


class ActionType(str):
    OBSERVE_VITALS   = "observe_vitals"
    ORDER_LABS       = "order_labs"
    ORDER_IMAGING    = "order_imaging"
    ASSIGN_TRIAGE    = "assign_triage"
    ALLOCATE_BED     = "allocate_bed"
    DISCHARGE        = "discharge"
    ADMIT            = "admit"
    ESCALATE         = "escalate"


# ── State ─────────────────────────────────────────────────────────────────────

class PatientState(BaseModel):
    patient_id: str
    wait_minutes: float
    chief_complaint: str
    age_group: Literal["infant", "child", "adult", "elderly"]
    gender: Literal["M", "F", "Other"]

    # Gated — only revealed after observe_vitals action
    heart_rate: Optional[int]         = None
    systolic_bp: Optional[int]        = None
    spo2: Optional[float]             = None
    respiratory_rate: Optional[int]   = None
    temperature: Optional[float]      = None

    # Gated — only revealed after order_labs / order_imaging
    labs_result: Optional[str]        = None
    imaging_result: Optional[str]     = None

    # Internal truth — NEVER exposed in observation
    true_esi: Optional[int]           = Field(None, exclude=True)
    assigned_esi: Optional[int]       = None
    is_dispositioned: bool            = False

    # Tracking for UI alerts
    source: str                       = "synthetic"
    has_deteriorated: bool            = False


class EnvironmentState(BaseModel):
    task_id: str
    sim_clock_minutes: float          = 0.0
    max_shift_minutes: float          = 360.0   # 6-hour shift
    action_tokens_remaining: int      = 10
    beds_available: int               = 12
    lab_slots_available: int          = 8
    imaging_slots_available: int      = 4
    patients: List[PatientState]      = []
    step_count: int                   = 0
    episode_done: bool                = False


# ── Action ────────────────────────────────────────────────────────────────────

class Action(BaseModel):
    patient_id: str = Field(..., description="ID of the patient to act on")
    action_type: str = Field(
        ...,
        description=(
            "One of: observe_vitals, order_labs, order_imaging, "
            "assign_triage, allocate_bed, discharge, admit, escalate"
        ),
    )
    esi_level: Optional[int] = Field(
        None, ge=1, le=5,
        description="Required when action_type is assign_triage (1=most severe)"
    )
    notes: Optional[str] = Field(None, description="Optional clinical reasoning")


# ── Observation ───────────────────────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    sim_clock_minutes: float
    action_tokens_remaining: int
    beds_available: int
    lab_slots_available: int
    imaging_slots_available: int
    step_count: int
    episode_done: bool
    patients: List[PatientState]
    last_action_feedback: str = ""


# ── Task / Grader API schemas ─────────────────────────────────────────────────

class TaskDescription(BaseModel):
    task_id: str
    description: str
    action_schema: dict
    observation_schema: dict
    max_steps: int = 50


class GraderRequest(BaseModel):
    task_id: str
    action: Action
    state: EnvironmentState


class GraderResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    done: bool
    next_state: EnvironmentState

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict = {}

class ResetResponse(BaseModel):
    observation: Observation


class BaselineRequest(BaseModel):
    task_id: str
    observation: Observation


class BaselineResponse(BaseModel):
    action: Action
    reasoning: str
