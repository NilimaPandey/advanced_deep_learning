def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    import json
    from pathlib import Path

    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    model = CoTModel()
    dataset = Dataset("train")
    results = []

    for idx in range(len(dataset)):
        question, correct_answer = dataset[idx]
        prompt = model.format_prompt(question)
        # batched_generate returns list of list when num_return_sequences is set
        generations_list = model.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature,
        )
        generations = generations_list[0]
        for gen in generations:
            parsed = model.parse_answer(gen)
            if is_answer_valid(parsed, correct_answer):
                results.append([question, correct_answer, gen])
                break

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
