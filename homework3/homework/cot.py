from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        HIGHLY OPTIMIZED VERSION - Tested patterns for unit conversion
        """
        messages = [
            {"role": "user", "content": "Convert 2 kg to g"},
            {"role": "assistant", "content": "2 kg = 2 × 1000 = 2000 g <answer>2000</answer>"},

            {"role": "user", "content": "Convert 5 m to cm"},
            {"role": "assistant", "content": "5 m = 5 × 100 = 500 cm <answer>500</answer>"},

            {"role": "user", "content": "Convert 3 km to m"},
            {"role": "assistant", "content": "3 km = 3 × 1000 = 3000 m <answer>3000</answer>"},

            {"role": "user", "content": "Convert 4 cm to mm"},
            {"role": "assistant", "content": "4 cm = 4 × 10 = 40 mm <answer>40</answer>"},

            {"role": "user", "content": "Convert 2 feet to inches"},
            {"role": "assistant", "content": "2 feet = 2 × 12 = 24 inches <answer>24</answer>"},

            {"role": "user", "content": "Convert 1.5 liters to ml"},
            {"role": "assistant", "content": "1.5 liters = 1.5 × 1000 = 1500 ml <answer>1500</answer>"},

            {"role": "user", "content": "Convert 3 yards to feet"},
            {"role": "assistant", "content": "3 yards = 3 × 3 = 9 feet <answer>9</answer>"},

            {"role": "user", "content": "Convert 2 pounds to ounces"},
            {"role": "assistant", "content": "2 pounds = 2 × 16 = 32 ounces <answer>32</answer>"},

            {"role": "user", "content": question}
        ]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})