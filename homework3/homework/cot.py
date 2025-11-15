from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        OPTIMIZED VERSION - Shorter, clearer examples that work with smaller models
        Focus on unit conversion patterns the model can easily learn
        """
        messages = [
            {"role": "user", "content": "How many grams in 2 kg?"},
            {"role": "assistant", "content": "2 kg × 1000 = 2000 g. <answer>2000</answer>"},

            {"role": "user", "content": "How many cm in 5 m?"},
            {"role": "assistant", "content": "5 m × 100 = 500 cm. <answer>500</answer>"},

            {"role": "user", "content": "How many m in 3 km?"},
            {"role": "assistant", "content": "3 km × 1000 = 3000 m. <answer>3000</answer>"},

            {"role": "user", "content": "How many mm in 4 cm?"},
            {"role": "assistant", "content": "4 cm × 10 = 40 mm. <answer>40</answer>"},

            {"role": "user", "content": "How many inches in 2 feet?"},
            {"role": "assistant", "content": "2 ft × 12 = 24 inches. <answer>24</answer>"},

            {"role": "user", "content": "How many feet in 3 yards?"},
            {"role": "assistant", "content": "3 yd × 3 = 9 feet. <answer>9</answer>"},

            {"role": "user", "content": "How many ml in 1.5 liters?"},
            {"role": "assistant", "content": "1.5 L × 1000 = 1500 ml. <answer>1500</answer>"},

            {"role": "user", "content": "How many ounces in 2 pounds?"},
            {"role": "assistant", "content": "2 lb × 16 = 32 oz. <answer>32</answer>"},

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