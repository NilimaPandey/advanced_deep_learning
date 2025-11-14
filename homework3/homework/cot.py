from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. We give the model
        one example and ask it to put the final numeric result inside
        <answer>...</answer>.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that solves unit conversion "
                    "questions step by step. Show brief reasoning and put the "
                    "final numeric result inside <answer>...</answer>."
                ),
            },
            # One-shot example
            {
                "role": "user",
                "content": "How many grams are in 2 kg?",
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 g. 2 × 1000 = <answer>2000</answer>",
            },
            # Actual question
            {
                "role": "user",
                "content": question,
            },
        ]

        # Use the tokenizer's chat template to format this as a single prompt string
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def load() -> BaseLLM:
    """
    Helper used by homework.__init__:
    returns an instance of the CoTModel.
    """
    return CoTModel()


def test_model():
    model = CoTModel()
    q = "How many centimeters are in 3 meters?"
    print(model.generate(q, max_new_tokens=40))
