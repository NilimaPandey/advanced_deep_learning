# cot.py - IMPROVED VERSION
from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        IMPROVED VERSION - Better examples, clearer reasoning format
        """
        messages = [
            {"role": "user", "content": "How many grams in 2 kilograms?"},
            {"role": "assistant", "content": "Let's think step by step. We know that 1 kg = 1000 g. So 2 kg = 2 × 1000 = 2000 grams. Therefore, the answer is <answer>2000</answer>"},

            {"role": "user", "content": "How many centimeters in 5 meters?"},
            {"role": "assistant", "content": "Let's think step by step. We know that 1 m = 100 cm. So 5 m = 5 × 100 = 500 centimeters. Therefore, the answer is <answer>500</answer>"},

            {"role": "user", "content": "How many meters in 3 kilometers?"},
            {"role": "assistant", "content": "Let's think step by step. We know that 1 km = 1000 m. So 3 km = 3 × 1000 = 3000 meters. Therefore, the answer is <answer>3000</answer>"},

            {"role": "user", "content": "How many millimeters in 7 centimeters?"},
            {"role": "assistant", "content": "Let's think step by step. We know that 1 cm = 10 mm. So 7 cm = 7 × 10 = 70 millimeters. Therefore, the answer is <answer>70</answer>"},

            {"role": "user", "content": "How many inches in 3 feet?"},
            {"role": "assistant", "content": "Let's think step by step. We know that 1 foot = 12 inches. So 3 feet = 3 × 12 = 36 inches. Therefore, the answer is <answer>36</answer>"},

            {"role": "user", "content": "How many feet in 4 yards?"},
            {"role": "assistant", "content": "Let's think step by step. We know that 1 yard = 3 feet. So 4 yards = 4 × 3 = 12 feet. Therefore, the answer is <answer>12</answer>"},

            {"role": "user", "content": "How many milliliters in 2 liters?"},
            {"role": "assistant", "content": "Let's think step by step. We know that 1 liter = 1000 ml. So 2 liters = 2 × 1000 = 2000 milliliters. Therefore, the answer is <answer>2000</answer>"},

            {"role": "user", "content": question}
        ]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )