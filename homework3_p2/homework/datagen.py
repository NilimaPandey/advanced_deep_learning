def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    import json
    from pathlib import Path

    from tqdm import tqdm

    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    # 1.7B-Instruct produces much better reasoning traces (per README recommendation)
    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    dataset = Dataset("train")
    results = []
    failures = 0

    for idx in tqdm(range(len(dataset)), desc="Generating RFT data"):
        question, correct_answer = dataset[idx]
        prompt = model.format_prompt(question)
        generations_list = model.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature,
        )
        generations = generations_list[0]
        found = False
        for gen in generations:
            parsed = model.parse_answer(gen)
            if is_answer_valid(parsed, correct_answer):
                results.append([question, correct_answer, gen])
                found = True
                break
        if not found:
            failures += 1

    print(f"Generated {len(results)}/{len(dataset)} examples ({failures} failures)")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
