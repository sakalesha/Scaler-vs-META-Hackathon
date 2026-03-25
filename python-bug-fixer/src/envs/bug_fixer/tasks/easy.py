TASK = {
    "id": "easy",
    "name": "Single Arithmetic Bug",
    "description": (
        "Fix a function that uses the wrong arithmetic operator. "
        "The bug is in one line and is clearly visible."
    ),
    "difficulty": "easy",
    "function_signature": "def add_numbers(a: int, b: int) -> int",
    "error_description": (
        "The function is supposed to return the sum of two numbers "
        "but it returns the wrong result for all inputs."
    ),
    "buggy_code": """\
def add_numbers(a: int, b: int) -> int:
    return a - b
""",
    "solution_code": """\
def add_numbers(a: int, b: int) -> int:
    return a + b
""",
    # basic tests: (args, expected_output)
    "basic_tests": [
        ((2, 3), 5),
        ((0, 0), 0),
        ((10, 5), 15),
    ],
    # edge case tests
    "edge_tests": [
        ((-1, -1), -2),
        ((-3, 5), 2),
        ((100, 0), 100),
    ],
}
