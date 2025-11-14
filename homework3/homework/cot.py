from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        # Use multiple examples (few-shot) with clear, short reasoning
        messages = [
            {
                "role": "user",
                "content": "How many grams are there per 2 kilograms?"
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 g, so 2 kg = 2000 g. <answer>2000</answer>"
            },
            {
                "role": "user",
                "content": "How many meters are there per 3 kilometers?"
            },
            {
                "role": "assistant",
                "content": "1 km = 1000 m, so 3 km = 3000 m. <answer>3000</answer>"
            },
            {
                "role": "user",
                "content": "How many centimeters are there per 5 meters?"
            },
            {
                "role": "assistant",
                "content": "1 m = 100 cm, so 5 m = 500 cm. <answer>500</answer>"
            },
            {
                "role": "user",
                "content": question
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt


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