import json
from pathlib import Path
from typing import List

from .cot import CoTModel
from .data import Dataset


def _is_correct_text(text: str, target: float, tol: float = 1e-3) -> bool:
    """Return True if parsed answer from text matches target within tolerance."""
    parser = CoTModel()
    pred = parser.parse_answer(text)
    if pred != pred:  # NaN
        return False
    return abs(pred - target) <= max(tol, tol * max(abs(target), abs(pred)))


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    """
    Generate an offline dataset for RFT.

    For each training question, we sample `oversample` chain-of-thought completions
    from CoTModel and keep one that yields the correct final answer (if any).
    The resulting dataset is written as JSON, where each row is:
        [question: str, answer: float, reasoning: str]
    """
    out_path = Path(output_json)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent.parent / out_path

    ds = Dataset("train")
    llm = CoTModel()

    collected: List[list] = []

    for q, y in ds.data:
        # Generate multiple samples for this question
        generations = llm.batched_generate(
            [q],
            max_new_tokens=128,
            temperature=temperature,
            num_return_sequences=oversample,
        )
        # `generations` is a flat list of length oversample
        chosen = None
        for g in generations:
            if _is_correct_text(g, y):
                chosen = g
                break
        if chosen is not None:
            collected.append([q, y, chosen])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(collected, f, indent=2)

    print(f"Wrote {len(collected)} examples to {out_path}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
