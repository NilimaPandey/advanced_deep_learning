from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Create a chat-style prompt with one chain-of-thought example.

        We instruct the model to solve unit conversion problems, show brief
        reasoning, and place the final numeric answer inside <answer>...</answer>.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that converts between units "
                    "(meters, feet, yards, kilometers, miles, kilograms, pounds, etc.). "
                    "Reason step by step briefly and put the final numeric answer "
                    "inside <answer>...</answer>. Be concise."
                ),
            },
            # One-shot example
            {
                "role": "user",
                "content": "How many grams are there in 2.5 kilograms?",
            },
            {
                "role": "assistant",
                "content": "1 kilogram = 1000 grams, so 2.5 × 1000 = <answer>2500</answer>.",
            },
            # Actual question
            {
                "role": "user",
                "content": question,
            },
        ]
        # Use the tokenizer's chat template for this model
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def load() -> BaseLLM:
    """Entry point used by homework.__init__.load_cot."""
    return CoTModel()


def test_model():
    model = CoTModel()
    q = "How many centimeters are there in 3 meters?"
    out = model.generate(q, max_new_tokens=64)
    print(out)
