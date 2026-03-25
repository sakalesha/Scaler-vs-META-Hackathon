""" BugFixAction, BugFixObservation """
""" It's like model schema as in backend """
from pydantic import BaseModel
from typing import Optional


class BugFixAction(BaseModel):
    """What the agent sends to the environment."""
    fixed_code: str                    # The corrected Python code
    explanation: str                   # What bug was fixed and why


class BugFixObservation(BaseModel):
    """What the environment sends back to the agent."""
    buggy_code: str                    # The broken Python code
    error_description: str             # Human-readable description of the bug
    function_signature: str            # e.g. "def add_numbers(a, b) -> int"
    task_id: str                       # "easy" / "medium" / "hard"
    step: int                          # Which step in the episode (0 = fresh reset)
    reward: Optional[float] = None     # Reward from last step (None on reset)
    done: bool = False                 # Is the episode over?
    feedback: Optional[str] = None     # What went wrong (if reward < 1.0)

""" Grader Returns a final score, test case breakdown, and feedback when code is submitted for evaluation. """
class GraderResult(BaseModel):
    """Returned by /grader endpoint."""
    score: float
    breakdown: dict
    feedback: str

""" When a user or agent queries what tasks are available, the server replies with this schema. """
class TaskInfo(BaseModel):
    """Single task description returned by /tasks endpoint."""
    id: str
    name: str
    description: str
    difficulty: str
    action_schema: dict

""" Structures the final report detailing the model, scores, and episodes per task when running a baseline model. """
class BaselineResult(BaseModel):
    """Returned by /baseline endpoint."""
    model: str
    scores: dict
    episodes_per_task: int