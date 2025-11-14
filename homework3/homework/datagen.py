def generate_dataset(output_json: str = "data/rft.json", oversample: int = 10, temperature: float = 0.6,
                     use_larger_model: bool = False):
    """
    Generate a dataset using rejection sampling / filtering.

    The idea is to:
    1. Load the CoT model
    2. Generate multiple answers for each question (oversample)
    3. Keep only the correct answers
    4. Save the filtered dataset in the format: [question, answer, reasoning]

    Args:
        output_json: Path to save the generated dataset (default: data/rft.json)
        oversample: Number of attempts per question
        temperature: Sampling temperature for generation
        use_larger_model: If True, use HuggingFaceTB/SmolLM2-1.7B-Instruct for better rollouts
    """
    import json
    from pathlib import Path

    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    # Load the CoT model - optionally use larger model for better generation
    if use_larger_model:
        print("Using larger model: HuggingFaceTB/SmolLM2-1.7B-Instruct")
        from .base_llm import BaseLLM
        model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    else:
        model = CoTModel()

    # Load the training dataset
    train_dataset = Dataset("train")

    generated_data = []
    success_count = 0

    print(f"Generating dataset with {oversample}x oversampling...")

    for idx in range(len(train_dataset)):
        question, correct_answer = train_dataset[idx]

        # Format the prompt
        prompt = model.format_prompt(question)

        # Generate multiple responses
        responses = model.batched_generate(
            [prompt] * oversample,
            temperature=temperature
        )

        # Filter for correct answers - keep the first correct one
        found_correct = False
        for response in responses:
            try:
                predicted_answer = model.parse_answer(response)

                # Check if the answer is correct
                if is_answer_valid(predicted_answer, correct_answer):
                    # Store in the format: [question, answer, reasoning]
                    # The reasoning includes the full response with <answer> tags
                    generated_data.append([
                        question,
                        correct_answer,
                        response
                    ])
                    found_correct = True
                    success_count += 1
                    break  # Only keep first correct answer per question
            except (ValueError, IndexError):
                # Skip invalid responses
                continue

        if (idx + 1) % 100 == 0:
            success_rate = success_count / (idx + 1) * 100
            print(f"Processed {idx + 1}/{len(train_dataset)} questions. Success rate: {success_rate:.1f}%")

    success_rate = success_count / len(train_dataset) * 100
    print(f"\nGenerated {len(generated_data)} valid samples from {len(train_dataset)} questions")
    print(f"Success rate: {success_rate:.1f}%")

    # Save the dataset
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w') as f:
        json.dump(generated_data, f, indent=2)

    print(f"Saved generated dataset to {output_json}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)