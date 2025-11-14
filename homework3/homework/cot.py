from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that solves unit conversions step by step. "
                        "Show your reasoning briefly and put the numeric result inside <answer>...</answer>. Be concise."},
            {"role": "user", "content": "How many grams are in 2 kg?"},
            {"role": "assistant", "content": "1 kg = 1000 g. 2 × 1000 = <answer>2000</answer>"},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def test_model():
    model = CoTModel()
    q = "How many centimeters are in 3 meters?"
    print(model.generate(q, max_new_tokens=40))
