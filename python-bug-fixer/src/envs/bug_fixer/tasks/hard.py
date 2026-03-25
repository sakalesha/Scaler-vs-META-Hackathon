TASK = {
    "id": "hard",
    "name": "Subtle Binary Search Bugs",
    "description": (
        "Fix a binary search function with 4 subtle bugs: "
        "wrong initial boundary, wrong loop condition, "
        "integer division error, and wrong pointer update."
    ),
    "difficulty": "hard",
    "function_signature": "def binary_search(arr: list, target: int) -> int",
    "error_description": (
        "The function is supposed to return the index of target in a sorted list, "
        "or -1 if not found. It currently returns wrong indices and "
        "sometimes loops infinitely."
    ),
    "buggy_code": """\
def binary_search(arr: list, target: int) -> int:
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) / 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    return -1
""",
    "solution_code": """\
def binary_search(arr: list, target: int) -> int:
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""",
    "basic_tests": [
        (([1, 3, 5, 7, 9], 5), 2),
        (([1, 3, 5, 7, 9], 1), 0),
        (([1, 3, 5, 7, 9], 9), 4),
    ],
    "edge_tests": [
        (([1, 3, 5, 7, 9], 4), -1),
        (([1], 1), 0),
        (([1, 2], 2), 1),
    ],
}
