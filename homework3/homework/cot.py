from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Implement the required in-context prompt using chat template.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that solves unit conversion "
                    "questions. Show short reasoning and put the final numeric "
                    "answer inside <answer></answer>."
                ),
            },
            {
                "role": "user",
                "content": "How many grams are in 2 kilograms?"
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 g, therefore 2×1000 = <answer>2000</answer>"
            },
            {
                "role": "user",
                "content": question
            }
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def load() -> BaseLLM:
    """
    REQUIRED BY THE GRADER.
    The grader calls: from homework import load_cot
    which is mapped to this function.
    """
    return CoTModel()
