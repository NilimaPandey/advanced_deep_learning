from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        BEST PERFORMING VERSION - 6 examples, short clear format
        This usually gets 20-30% accuracy (12-18 points)
        """
        messages = [
            {"role": "user", "content": "How many g in 2 kg?"},
            {"role": "assistant", "content": "2 kg = 2 * 1000 = 2000 g. <answer>2000</answer>"},

            {"role": "user", "content": "How many cm in 5 m?"},
            {"role": "assistant", "content": "5 m = 5 * 100 = 500 cm. <answer>500</answer>"},

            {"role": "user", "content": "How many m in 3 km?"},
            {"role": "assistant", "content": "3 km = 3 * 1000 = 3000 m. <answer>3000</answer>"},

            {"role": "user", "content": "How many mm in 4 cm?"},
            {"role": "assistant", "content": "4 cm = 4 * 10 = 40 mm. <answer>40</answer>"},

            {"role": "user", "content": "How many inches in 2 feet?"},
            {"role": "assistant", "content": "2 feet = 2 * 12 = 24 inches. <answer>24</answer>"},

            {"role": "user", "content": "How many feet in 5 yards?"},
            {"role": "assistant", "content": "5 yards = 5 * 3 = 15 feet. <answer>15</answer>"},

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