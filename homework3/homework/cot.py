from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Build a chat dialogue with:
        - Short system instructions
        - One worked example
        - The user's question

        We then turn this into a single prompt string using the tokenizer's
        chat template. The grader expects you to use apply_chat_template
        with add_generation_prompt=True, tokenize=False.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that solves unit conversion "
                    "problems (meters, feet, yards, kilometers, miles, "
                    "kilograms, pounds, etc.). "
                    "Reason step by step briefly and put the final numeric "
                    "result inside <answer>...</answer>. Be concise."
                ),
            },
            # One-shot example
            {
                "role": "user",
                "content": "How many grams are there in 2.5 kilograms?",
            },
            {
                "role": "assistant",
                "content": (
                    "1 kilogram = 1000 grams, so "
                    "2.5 × 1000 = <answer>2500</answer>."
                ),
            },
            # Actual question
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


def load() -> BaseLLM:
    """
    Entry point used by homework.__init__:
    the grader does `from homework import load_cot`
    which redirects here.
    """
    return CoTModel()


def test_model():
    model = CoTModel()
    q = "How many centimeters are there in 3 meters?"
    out = model.generate(q, max_new_tokens=64)
    print(out)
