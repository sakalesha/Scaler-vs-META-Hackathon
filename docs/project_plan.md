# Medical Triage Advisor — OpenEnv Hackathon Project Plan

---

## 1. Project Overview

**Environment Name:** medical-triage-advisor
**Domain:** Emergency Medicine — Emergency Severity Index (ESI) Triage
**Core Task:** Agent receives a patient case (vitals + symptoms + history) and must correctly assign an ESI triage level (1–5), provide clinical reasoning, recommend disposition, and suggest immediate interventions.
**Why This Wins:** No medical RL environment exists in OpenEnv. ESI is a real, globally standardized protocol. Grading is deterministic. Partial credit is clinically meaningful. Cannot be gamed by pattern matching.

---

## 2. Pre-Implementation Decisions (Resolve Before Writing Code)

These must be locked in before a single line of implementation is written. Changing these mid-build breaks everything.

---

### 2.1 ESI Level Ground Truth Strategy

**Decision needed:** How do we define "correct" ESI for each patient case?

**Options:**
- Option A: Single correct ESI level (strict) — agent must match exactly or be off by 1
- Option B: Acceptable range — e.g., ESI 1 or 2 both acceptable for borderline cases
- Option C: Weighted correct — ESI 2 gets 0.8 reward when correct is ESI 1

**Recommended:** Option C — matches real clinical reality where borderline cases exist.

**Decision:** Use Option C — weighted scoring with acceptable adjacency.

---

### 2.2 Grader Architecture — What Gets Scored

**Decision needed:** What exactly does the grader measure?

**Locked scoring breakdown:**

| Component | Weight | How measured |
|---|---|---|
| ESI level (exact) | 0.40 | Exact match = 0.40, off by 1 = 0.20, off by 2+ = 0.00 |
| Clinical reasoning | 0.25 | Keyword matching against critical finding list |
| Disposition | 0.20 | Must match expected disposition room |
| Immediate interventions | 0.15 | Must mention at least N correct interventions |
| **Total** | **1.00** | Always clamped to [0.0, 1.0] |

**Undertriage penalty:** If agent assigns ESI 4 or 5 when correct is ESI 1 or 2, score is capped at 0.10 regardless of reasoning quality. This mirrors real clinical ethics.

---

### 2.3 Patient Case Design

**Decision needed:** How many patient cases per difficulty level?

**Recommended:** 3 cases per level (9 total) — environment randomly selects one per episode. This prevents memorization and adds variance to the score check in Phase 2.

**Case selection:** Seeded random per episode so baseline.py remains reproducible.

**Case structure per task file:**
```
task_id: str
difficulty: str
patient: { age, sex, arrival_mode }
chief_complaint: str
vitals: { bp_systolic, bp_diastolic, hr, rr, temp, spo2, gcs }
history: str
correct_esi: int
acceptable_esi: list[int]
correct_disposition: str
critical_findings: list[str]      # keywords grader checks in reasoning
required_interventions: list[str] # at least 2 of these must appear
explanation: str                  # why this ESI — for README documentation
```

---

### 2.4 Action Schema — What the Agent Must Send

**Locked fields:**

```
esi_level: int               # 1, 2, 3, 4, or 5 — required
reasoning: str               # clinical justification — required
disposition: str             # one of 5 fixed options — required
immediate_interventions: str # free text of what to do now — required
```

**Disposition options (fixed vocabulary):**
- "resuscitation_bay"
- "acute_care"
- "fast_track"
- "waiting_room"
- "discharge"

**Why fixed vocabulary for disposition:** Prevents the agent from inventing terms that are semantically correct but don't match. Keeps grading deterministic.

---

### 2.5 Episode Structure

**Decision needed:** How many steps per episode? Single-turn or multi-turn?

**Recommended:** Single-turn per episode, 1 step max.

**Reasoning:** Triage is a real-time decision under pressure. A real triage nurse doesn't get to revise their ESI. Single-turn also keeps the grader simple and deterministic.

**Episode lifecycle:**
```
reset(task_id) → observation (patient case)
     ↓
step(action)   → observation (with reward + feedback + done=True)
     ↓
Episode ends. Call reset() for new case.
```

---

### 2.6 Session Management

**Decision needed:** How do we track sessions server-side?

**Approach:** In-memory dictionary mapping `session_id → environment instance`.

**session_id:** UUID4 generated at reset(), returned to client, required for step() and state() calls.

**Session expiry:** Sessions are not persisted. Restart of server clears all sessions. This is fine for the hackathon scope.

---

### 2.7 Baseline Model Selection

**Decision needed:** Which model for baseline.py?

**Recommended:** `Qwen/Qwen2.5-0.5B-Instruct`

**Why:**
- Small enough to run on free hardware
- Instruction-tuned — follows structured prompts
- Judges can reproduce on their machines without GPU
- Known to score non-trivially on reasoning tasks (won't get 0.0 on easy)

**Seed:** 42 everywhere. Greedy decoding (do_sample=False) for full determinism.

---

### 2.8 Dependencies — Locked List

```
# Runtime (goes in Dockerfile + pyproject.toml)
fastapi==0.115.0
uvicorn==0.30.0
pydantic==2.7.0
openenv-core          # latest stable

# Baseline script only (not in Dockerfile)
transformers==4.44.0
torch==2.3.0
accelerate==0.31.0

# Testing
pytest==8.2.0
httpx==0.27.0         # for async test client
```

**Note:** torch is NOT in the Dockerfile — it makes the image 5GB+. Baseline runs locally, not inside the container.

---

## 3. Technical Architecture

```
Client (training script / judge's machine)
        │
        │  HTTP POST /reset?task_id=easy
        │  HTTP POST /step  { session_id, action }
        │  HTTP GET  /state?session_id=...
        │  HTTP GET  /tasks
        │  HTTP POST /grader  { session_id, task_id }
        │  HTTP POST /baseline
        │
        ▼
FastAPI Server (server.py)
        │
        ├── Session store: { uuid → BugFixerEnvironment }
        │
        ├── BugFixerEnvironment (environment.py)
        │       ├── reset()   → loads patient case, clears state
        │       ├── step()    → calls grader, returns observation
        │       └── state()   → returns episode metadata
        │
        ├── Grader (grader.py)
        │       ├── grade_esi()           → 0.0–0.40
        │       ├── grade_reasoning()     → 0.0–0.25
        │       ├── grade_disposition()   → 0.0–0.20
        │       └── grade_interventions() → 0.0–0.15
        │
        └── Tasks (tasks/)
                ├── easy.py    → 3 patient cases, ESI 1–2
                ├── medium.py  → 3 patient cases, ESI 2–3 borderline
                └── hard.py    → 3 patient cases, subtle multi-factor
```

---

## 4. Difficulty Curve Design

**Target scores for Qwen 0.5B (baseline model):**

| Task | Target Baseline Score | Why |
|---|---|---|
| Easy | 0.55 – 0.75 | Obvious vitals, model should mostly get ESI right |
| Medium | 0.25 – 0.50 | Needs clinical context reasoning, model will partially succeed |
| Hard | 0.05 – 0.25 | Subtle multi-factor, model likely misses key connections |

**Target scores for Nemotron 3 Super (judge's model):**

| Task | Target Score | Why |
|---|---|---|
| Easy | 0.70 – 0.90 | Strong model should mostly nail obvious presentations |
| Medium | 0.40 – 0.65 | Should get ESI right but miss some reasoning keywords |
| Hard | 0.15 – 0.40 | Should struggle — this is the point |

**If Nemotron scores >0.85 on hard → hard task is too easy. Must revise.**

---

## 5. Patient Case Design (9 Cases Total)

### Easy Cases (ESI 1–2, obvious presentations)
- Case E1: Cardiac arrest — unresponsive, no pulse → ESI 1
- Case E2: Severe respiratory distress — SpO2 82%, accessory muscle use → ESI 1
- Case E3: Active seizure, still seizing on arrival → ESI 1

### Medium Cases (ESI 2–3, ambiguous)
- Case M1: "Worst headache of life" + blood thinners → ESI 2 (subarachnoid bleed risk)
- Case M2: Chest pain + diabetic + diaphoresis, vitals borderline → ESI 2 (NSTEMI risk)
- Case M3: Abdominal pain + elderly + fever, vitals near normal → ESI 2–3 (sepsis risk)

### Hard Cases (ESI 2, subtle multi-factor)
- Case H1: DVT + new dyspnea + OCP + long flight → ESI 2 (PE risk)
- Case H2: Mild confusion + elderly + UTI symptoms, afebrile → ESI 2 (urosepsis risk)
- Case H3: Young athlete, "racing heart", syncope once, now fine → ESI 2 (arrhythmia risk)

---

## 6. Exploit Prevention

**Known exploits to prevent in grader design:**

| Exploit | Prevention |
|---|---|
| Always submit ESI 1 (highest urgency) | Overtriage penalty: ESI 1 on non-critical = cap at 0.30 |
| Dump all medical keywords in reasoning | Keywords must appear in coherent sentences (context check) |
| Always say "resuscitation_bay" | Overtriage on disposition also penalized |
| Submit empty/gibberish code | Grader returns 0.0 immediately on empty strings |
| Repeat same action every step | Episode is single-turn — no benefit to repeating |

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| HF Space sleeps before judging | High | Disqualification | Add /health ping, set Space to always-on if possible |
| Baseline scores don't reproduce | Medium | Disqualification | Seed everywhere, greedy decoding, pin all versions |
| Grader returns same score for all inputs | Medium | Phase 2 fail | Test grader with 10+ different responses manually |
| Docker build fails at judge's machine | Low | Disqualification | Test with --no-cache before submitting |
| Hard task too easy for Nemotron | Medium | Low novelty score | Design H3 with 4+ factors required for correct answer |
| Patient cases have medical errors | Low | Credibility loss | Cross-check all ESI assignments against official ESI handbook |

---

## 8. Pre-Implementation Checklist

### Design Decisions (Must be done BEFORE coding)
- [ ] Confirm all 9 patient cases are medically accurate
- [ ] Confirm ESI assignments against official ESI v4 handbook
- [ ] Lock grader weights (40/25/20/15 split confirmed)
- [ ] Lock disposition vocabulary (5 fixed options confirmed)
- [ ] Confirm episode structure (single-turn, 1 step max)
- [ ] Confirm baseline model (Qwen 0.5B, greedy, seed 42)
- [ ] Confirm all dependency versions

### Environment (implement in this order)
- [ ] models.py — Vitals, TriageObservation, TriageAction, GraderResult, TaskInfo, BaselineResult
- [ ] tasks/easy.py — 3 patient cases with all fields
- [ ] tasks/medium.py — 3 patient cases with all fields
- [ ] tasks/hard.py — 3 patient cases with all fields
- [ ] grader.py — grade_esi, grade_reasoning, grade_disposition, grade_interventions, grade()
- [ ] environment.py — reset(), step(), state(), session state management
- [ ] server/server.py — all 7 endpoints wired up

### Scripts
- [ ] baseline.py — runs Qwen on all 3 tasks, fixed seeds, prints scores
- [ ] tests/test_env.py — test reset, step, grader range, all tasks, determinism

### Config & Deployment
- [ ] openenv.yaml — valid structure, all tasks listed
- [ ] pyproject.toml — all runtime dependencies pinned
- [ ] server/Dockerfile — builds clean with --no-cache
- [ ] README.md — description, action/obs spaces, setup, baseline scores table

### Validation (before submitting)
- [ ] Run: openenv validate
- [ ] Run: docker build --no-cache -t triage .
- [ ] Run: docker run -p 8000:8000 triage
- [ ] Run: curl http://localhost:8000/health → 200
- [ ] Run: curl http://localhost:8000/tasks → all 3 tasks listed
- [ ] Run: python baseline.py → scores print, no errors
- [ ] Run: curl -X POST http://localhost:8000/baseline → matches baseline.py output
- [ ] Run: pytest tests/ → all pass
- [ ] Deploy to HF Space → ping public URL → 200
- [ ] Test reset() on HF Space URL from local machine

---

## 9. Build Order (File by File)

```
Day 1 — Core Logic
  1. models.py
  2. tasks/easy.py
  3. tasks/medium.py
  4. tasks/hard.py
  5. grader.py
  6. environment.py

Day 2 — Server + Scripts
  7. server/server.py
  8. baseline.py
  9. tests/test_env.py

Day 3 — Packaging + Deployment
  10. openenv.yaml
  11. pyproject.toml
  12. server/Dockerfile
  13. README.md
  14. Docker test
  15. HF Space deploy
  16. Final validation run
```

---

## 10. Things Still Needed (Open Items)

- [ ] **HF Account:** Must have a Hugging Face account to push the Space. Username needed for openenv.yaml author field.
- [ ] **HF Space created:** Create an empty Space at huggingface.co before Day 3 so the URL is known for README.
- [ ] **openenv validate tool:** Install and confirm it runs on your machine before Day 3.
- [ ] **Docker Desktop:** Must be installed and running before Day 3.
- [ ] **Medical review:** All 9 patient cases should be cross-checked against ESI v4 Implementation Handbook (free PDF online) before finalizing tasks/.
- [ ] **Hackathon submission URL:** Know where and how to submit before the deadline.
- [ ] **Submission deadline:** Confirm exact deadline time and timezone.
