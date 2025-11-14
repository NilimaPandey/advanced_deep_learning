from .base_llm import BaseLLM


# in cot.py
from .base_llm import BaseLLM

class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Chat-style prompt that makes the model behave like a strict
        unit-conversion calculator. It gives a single example and
        forces the output format <answer>NUMBER</answer>.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a calculator for unit conversions. "
                    "For each question, compute the correct numeric answer "
                    "and reply with EXACTLY one line:\n"
                    "<answer>NUMBER</answer>\n"
                    "No words, no explanation, no units."
                ),
            },
            {
                "role": "user",
                "content": "How many grams are there in 2.5 kilograms?",
            },
            {
                "role": "assistant",
                "content": "<answer>2500</answer>",
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
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