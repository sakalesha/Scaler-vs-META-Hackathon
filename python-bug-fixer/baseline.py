import random
import torch
import numpy as np
from transformers import pipeline

from src.envs.bug_fixer.environment import BugFixerEnvironment

# Fix seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
EPISODES_PER_TASK = 5
TASK_IDS = ["easy", "medium", "hard"]


def run_baseline() -> dict:
    """
    Evaluates a baseline LLM (Qwen) across all available tasks.
    Instantiates the environment, prompts the model with the buggy code,
    extracts the response, and grades the fixed code.
    """
    # Initialize the Hugging Face text-generation pipeline
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        max_new_tokens=512,
        do_sample=False,   # greedy decoding for deterministic results
    )

    scores = {}

    # Iterate through every task (easy, medium, hard)
    for task_id in TASK_IDS:
        task_scores = []

        # Run multiple episodes to get an average score (in greedy it will be the same, but good practice)
        for episode in range(EPISODES_PER_TASK):
            env = BugFixerEnvironment(task_id=task_id)
            obs = env.reset()

            # Construct the prompt telling the LLM what to do
            prompt = f"""You are a Python bug fixer.

Here is broken Python code:
{obs.buggy_code}

Error description: {obs.error_description}
Function signature: {obs.function_signature}

Return ONLY the fixed Python code with no explanation outside the code block."""

            # Ask the model to generate the fix
            output = generator(prompt)[0]["generated_text"]

            # Extract the pure Python code from markdown code blocks in the model's output
            fixed_code = output.strip()
            if "```python" in fixed_code:
                fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
            elif "```" in fixed_code:
                fixed_code = fixed_code.split("```")[1].split("```")[0].strip()

            # Submit the extracted code back to the environment
            # We use a mocked object that mimics the BugFixAction pydantic model
            result = env.step(
                type('BugFixAction', (), {
                    'fixed_code': fixed_code,
                    'explanation': 'baseline run'
                })()
            )
            # Save the reward (score) for this attempt
            task_scores.append(result.reward or 0.0)

        scores[task_id] = round(sum(task_scores) / len(task_scores), 4)
        print(f"  {task_id:8s} → avg score: {scores[task_id]:.4f}")

    return {
        "model": MODEL_NAME,
        "scores": scores,
        "episodes_per_task": EPISODES_PER_TASK,
    }


if __name__ == "__main__":
    print(f"Running baseline with {MODEL_NAME}...\n")
    results = run_baseline()
    print("\nFinal Scores:")
    for task_id, score in results["scores"].items():
        print(f"  {task_id}: {score}")
