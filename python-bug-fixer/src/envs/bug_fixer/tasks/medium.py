TASK = {
    "id": "medium",
    "name": "Multiple Bugs in Loop",
    "description": (
        "Fix a function with two bugs: a wrong initial value "
        "and an off-by-one error in the loop range."
    ),
    "difficulty": "medium",
    "function_signature": "def find_max(nums: list) -> int",
    "error_description": (
        "The function is supposed to return the maximum value in a list "
        "but fails for lists with all negative numbers and crashes on "
        "the last iteration."
    ),
    "buggy_code": """\
def find_max(nums: list) -> int:
    max_val = 0
    for i in range(len(nums) + 1):
        if nums[i] > max_val:
            max_val = nums[i]
    return max_val
""",
    "solution_code": """\
def find_max(nums: list) -> int:
    max_val = nums[0]
    for i in range(len(nums)):
        if nums[i] > max_val:
            max_val = nums[i]
    return max_val
""",
    "basic_tests": [
        (([1, 3, 2],), 3),
        (([5, 5, 5],), 5),
        (([1],), 1),
    ],
    "edge_tests": [
        (([-1, -5, -2],), -1),
        (([0, -1, -2],), 0),
        (([100, 1, 50],), 100),
    ],
}
