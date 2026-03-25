""" compute_reward() """

import ast
import traceback
from typing import Tuple


def run_tests(fixed_code: str, tests: list, func_name: str) -> float:
    """
    Run a list of (args, expected) tests against fixed_code.
    Returns fraction of tests passed (0.0 to 1.0).
    """
    if not tests:
        return 0.0

    namespace = {}
    try:
        exec(fixed_code, namespace)
    except Exception:
        return 0.0

    func = namespace.get(func_name)
    if func is None:
        return 0.0

    passed = 0
    for args, expected in tests:
        try:
            result = func(*args)
            if result == expected:
                passed += 1
        except Exception:
            pass

    return passed / len(tests)


def grade(fixed_code: str, task: dict) -> Tuple[float, dict, str]:
    """
    Grade the agent's fixed code against a task.
    Returns (score, breakdown, feedback).
    Score is always in [0.0, 1.0].
    """
    breakdown = {
        "parses": 0.0,
        "executes": 0.0,
        "basic_tests": 0.0,
        "edge_tests": 0.0,
    }
    feedback_parts = []

    # Step 1 — Does it parse? (0.2)
    try:
        ast.parse(fixed_code)
        breakdown["parses"] = 0.2
    except SyntaxError as e:
        feedback = f"Syntax error: {e}"
        return 0.0, breakdown, feedback

    # Step 2 — Does it execute? (0.2)
    namespace = {}
    try:
        exec(fixed_code, namespace)
        breakdown["executes"] = 0.2
    except Exception as e:
        feedback = f"Runtime error on load: {e}"
        score = breakdown["parses"]
        return round(score, 4), breakdown, feedback

    # Extract function name from signature
    func_name = task["function_signature"].split("(")[0].replace("def ", "").strip()

    # Step 3 — Basic tests (0.3)
    basic_ratio = run_tests(fixed_code, task["basic_tests"], func_name)
    breakdown["basic_tests"] = round(0.3 * basic_ratio, 4)
    if basic_ratio < 1.0:
        feedback_parts.append(
            f"Passed {int(basic_ratio * len(task['basic_tests']))}"
            f"/{len(task['basic_tests'])} basic tests"
        )

    # Step 4 — Edge tests (0.3)
    edge_ratio = run_tests(fixed_code, task["edge_tests"], func_name)
    breakdown["edge_tests"] = round(0.3 * edge_ratio, 4)
    if edge_ratio < 1.0:
        feedback_parts.append(
            f"Passed {int(edge_ratio * len(task['edge_tests']))}"
            f"/{len(task['edge_tests'])} edge case tests"
        )

    score = sum(breakdown.values())
    score = round(max(0.0, min(1.0, score)), 4)

    feedback = (
        " | ".join(feedback_parts)
        if feedback_parts
        else "All tests passed!"
    )

    return score, breakdown, feedback