import json
from pathlib import Path
from .cot import CoTModel
from .data import Dataset


def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    model = CoTModel()
    data = Dataset("train").data

    results = []

    for q, y in data:
        # generate N reasoning samples
        outs = model.batched_generate(
            [q],
            num_return_sequences=oversample,
            temperature=temperature,
        )
        for g in outs:
            pred = model.parse_answer(g)
            if pred == pred and abs(pred - y) <= 1e-3 * max(1, abs(y)):
                results.append([q, y, g])
                break

    path = Path(output_json)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / path

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} entries to {path}")
