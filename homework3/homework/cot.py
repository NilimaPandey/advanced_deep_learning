from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that solves unit conversion problems. Be concise and show your work step by step."
            },
            {
                "role": "user",
                "content": "How many meters are there per 5 kilometers?"
            },
            {
                "role": "assistant",
                "content": "1 kilometer = 1000 meters. So 5 kilometers = 5 × 1000 = 5000 meters. <answer>5000</answer>"
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