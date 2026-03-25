""" reset(), step(), state() """

from .models import BugFixAction, BugFixObservation
from .grader import grade
from .tasks.easy import TASK as EASY_TASK
from .tasks.medium import TASK as MEDIUM_TASK
from .tasks.hard import TASK as HARD_TASK

TASKS = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}

MAX_STEPS = 3   # Agent gets 3 attempts per episode


class BugFixerEnvironment:

    def __init__(self, task_id: str = "easy"):
        assert task_id in TASKS, f"Unknown task: {task_id}"
        self.task_id = task_id
        self.task = TASKS[task_id]
        self.step_count = 0
        self.last_score = 0.0
        self.last_breakdown = {}
        self.last_feedback = ""
        self.done = False

    def reset(self) -> BugFixObservation:
        self.step_count = 0
        self.last_score = 0.0
        self.last_breakdown = {}
        self.last_feedback = ""
        self.done = False

        return BugFixObservation(
            buggy_code=self.task["buggy_code"],
            error_description=self.task["error_description"],
            function_signature=self.task["function_signature"],
            task_id=self.task_id,
            step=0,
            reward=None,
            done=False,
            feedback=None,
        )

    def step(self, action: BugFixAction) -> BugFixObservation:
        if self.done:
            # Episode already over — return terminal state
            return BugFixObservation(
                buggy_code=self.task["buggy_code"],
                error_description=self.task["error_description"],
                function_signature=self.task["function_signature"],
                task_id=self.task_id,
                step=self.step_count,
                reward=0.0,
                done=True,
                feedback="Episode already completed. Call reset() to start again.",
            )

        self.step_count += 1

        # Grade the submission
        score, breakdown, feedback = grade(action.fixed_code, self.task)
        self.last_score = score
        self.last_breakdown = breakdown
        self.last_feedback = feedback

        # Episode ends if: perfect score OR max steps reached
        self.done = (score == 1.0) or (self.step_count >= MAX_STEPS)

        return BugFixObservation(
            buggy_code=self.task["buggy_code"],
            error_description=self.task["error_description"],
            function_signature=self.task["function_signature"],
            task_id=self.task_id,
            step=self.step_count,
            reward=score,
            done=self.done,
            feedback=feedback,
        )

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step_count": self.step_count,
            "max_steps": MAX_STEPS,
            "last_score": self.last_score,
            "last_breakdown": self.last_breakdown,
            "done": self.done,
        }
