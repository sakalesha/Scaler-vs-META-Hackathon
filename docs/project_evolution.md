# Project Evolution Log

This document tracks the major milestones, design shifts, and key decisions made during the development of the Medical Triage Advisor.

## 2026-03-31: Project Kickoff & MVP Definition
- **Decision:** Shift from a full-scale clinical simulation to a "Walking Skeleton" MVP first.
- **Goal:** Get the core FastAPI + OpenEnv loop working with a single life-threatening case (ESI 1).
- **Structure:** Established the `/docs` folder for better project organization.
- **Initial Plan:** Moved the raw `PROJECT_PLAN (1).md` to a cleaner `/docs/project_plan.md`.

## Next Milestones
- [ ] Implement `models.py` (The data foundation).
- [ ] Create the first patient case (Cardiac Arrest - ESI 1).
- [ ] Build the basic `grader.py` (ESI level check only).
- [ ] Deploy a test server to verify the API response.
