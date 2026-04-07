"""
Grader for ClinicalTriageEnv-v0
Returns a score in [0.0, 1.0] based on how well the action fits
the clinical situation. Score components:

  - Triage accuracy      (assign_triage)
  - Escalation urgency   (escalate on ESI-1/2)
  - Information value    (observe before acting)
  - Resource efficiency  (don't waste labs on ESI-5)
  - Throughput           (disposition when ready)
  - Critical delay       (penalty for ignoring ESI-1)
"""

from models import TriageAction, EnvironmentState, GraderResponse
from environment import apply_action


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _triage_accuracy_score(assigned: int, true: int) -> float:
    """Graded accuracy: exact=1.0, off-by-one=0.65, off-by-two=0.35, worse=0.05"""
    diff = abs(assigned - true)
    return {0: 1.0, 1: 0.65, 2: 0.35}.get(diff, 0.05)


def _has_gathered_info(patient) -> bool:
    """Did the agent observe vitals before acting?"""
    return patient.heart_rate is not None


def _critical_patient_waiting(state: EnvironmentState) -> bool:
    """Is there an un-triaged ESI-1 patient still waiting?"""
    return any(
        p.true_esi == 1 and not p.is_dispositioned and p.assigned_esi is None
        for p in state.patients
    )


# ── Main grader ───────────────────────────────────────────────────────────────

def grade(request_data: dict) -> GraderResponse:
    from models import GraderRequest
    req = GraderRequest(**request_data)

    state  = req.state
    action = req.action

    patient = next(
        (p for p in state.patients if p.patient_id == action.patient_id), None
    )

    # ── Invalid action ────────────────────────────────────────────────────────
    if patient is None:
        new_state, _ = apply_action(state, action)
        return GraderResponse(
            score=0.05,
            rationale="Patient not found — invalid action target.",
            done=new_state.episode_done,
            next_state=new_state,
        )

    if patient.is_dispositioned:
        new_state, _ = apply_action(state, action)
        return GraderResponse(
            score=0.05,
            rationale="Patient already dispositioned — wasted action.",
            done=new_state.episode_done,
            next_state=new_state,
        )

    score     = 0.0
    rationale = ""
    atype     = action.action_type
    true_esi  = patient.true_esi or 3
    has_info  = _has_gathered_info(patient)

    # ── assign_triage ─────────────────────────────────────────────────────────
    if atype == "assign_triage":
        if action.esi_level is None:
            score     = 0.05
            rationale = "assign_triage called without esi_level."
        else:
            base  = _triage_accuracy_score(action.esi_level, true_esi)
            bonus = 0.10 if has_info else -0.10   # reward info-gathering first
            score = min(1.0, max(0.05, base + bonus))
            rationale = (
                f"Assigned ESI-{action.esi_level} vs true ESI-{true_esi}. "
                f"Accuracy score {base:.2f}. "
                + ("Info gathered before triaging (+0.10)." if has_info
                   else "No vitals observed before triaging (−0.10).")
            )

    # ── escalate ──────────────────────────────────────────────────────────────
    elif atype == "escalate":
        if true_esi == 1:
            score     = 1.0
            rationale = "Correct escalation of ESI-1 patient — highest priority action."
        elif true_esi == 2:
            score     = 0.80
            rationale = "Escalation of ESI-2 patient — slightly aggressive but appropriate."
        elif true_esi == 3:
            score     = 0.30
            rationale = "Escalation of stable ESI-3 patient — over-triage, wastes resus resources."
        else:
            score     = 0.05
            rationale = "Escalation of non-urgent patient — major resource waste."

    # ── observe_vitals ────────────────────────────────────────────────────────
    elif atype == "observe_vitals":
        if has_info:
            score     = 0.15
            rationale = "Vitals already observed for this patient — redundant action."
        elif true_esi <= 2:
            score     = 0.90
            rationale = "Observing vitals on high-acuity patient — correct prioritisation."
        elif true_esi == 3:
            score     = 0.75
            rationale = "Observing vitals on urgent patient — appropriate."
        else:
            score     = 0.50
            rationale = "Observing vitals on low-acuity patient — low value use of token."

    # ── order_labs ────────────────────────────────────────────────────────────
    elif atype == "order_labs":
        if not has_info:
            score     = 0.25
            rationale = "Labs ordered without checking vitals first — poor clinical reasoning."
        elif true_esi <= 2:
            score     = 0.85
            rationale = "Labs on high-acuity patient with vitals known — clinically appropriate."
        elif true_esi == 3:
            score     = 0.70
            rationale = "Labs on urgent patient — reasonable."
        else:
            score     = 0.20
            rationale = "Labs on low-acuity patient — likely unnecessary, wastes slot."

    # ── order_imaging ─────────────────────────────────────────────────────────
    elif atype == "order_imaging":
        if not has_info:
            score     = 0.20
            rationale = "Imaging ordered without vitals — poor information gathering order."
        elif true_esi == 1:
            score     = 0.90
            rationale = "Imaging on critical patient — high diagnostic value."
        elif true_esi == 2:
            score     = 0.75
            rationale = "Imaging on emergent patient — appropriate."
        elif true_esi == 3:
            score     = 0.55
            rationale = "Imaging on urgent patient — acceptable."
        else:
            score     = 0.10
            rationale = "Imaging on non-urgent patient — wastes expensive slot."

    # ── discharge / admit ─────────────────────────────────────────────────────
    elif atype in ("discharge", "admit"):
        if patient.assigned_esi is None:
            score     = 0.20
            rationale = f"Patient dispositioned without triage assignment — process violation."
        elif atype == "discharge" and patient.assigned_esi >= 4:
            score     = 0.90
            rationale = "Appropriate discharge of low-acuity patient."
        elif atype == "admit" and patient.assigned_esi <= 2:
            score     = 0.90
            rationale = "Correct admission of high-acuity patient."
        elif atype == "discharge" and patient.assigned_esi <= 2:
            score     = 0.05
            rationale = "DANGEROUS: High-acuity patient discharged — critical error."
        else:
            score     = 0.55
            rationale = "Disposition completed with moderate acuity match."

    # ── allocate_bed ──────────────────────────────────────────────────────────
    elif atype == "allocate_bed":
        if state.beds_available == 0:
            score     = 0.05
            rationale = "Tried to allocate bed when none available."
        elif true_esi <= 2:
            score     = 0.85
            rationale = "Bed allocated for high-acuity patient — correct."
        else:
            score     = 0.50
            rationale = "Bed allocated for lower-acuity patient — acceptable."

    else:
        score     = 0.10
        rationale = f"Unknown action type: {atype}"

    # ── Critical delay penalty ────────────────────────────────────────────────
    if _critical_patient_waiting(state) and atype not in ("escalate", "observe_vitals", "assign_triage"):
        non_critical_patient = (patient.true_esi or 3) >= 3
        if non_critical_patient:
            score     = max(0.05, score - 0.25)
            rationale += " ⚠️ ESI-1 patient waiting untouched — penalty applied."

    new_state, feedback = apply_action(state, action)
    rationale += f" | {feedback}"

    return GraderResponse(
        score=round(score, 4),
        rationale=rationale,
        done=new_state.episode_done,
        next_state=new_state,
    )
