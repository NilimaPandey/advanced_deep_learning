from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        # Add system message with clear instruction
        messages = [
            {
                "role": "system",
                "content": "Answer unit conversion questions. Show calculation and put final number in <answer>NUMBER</answer> tags."
            },
            {
                "role": "user",
                "content": "How many grams are in 5 kilograms?"
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 g, so 5 kg = 5 * 1000 = 5000 g. <answer>5000</answer>"
            },
            {
                "role": "user",
                "content": "How many centimeters are in 2 meters?"
            },
            {
                "role": "assistant",
                "content": "1 m = 100 cm, so 2 m = 2 * 100 = 200 cm. <answer>200</answer>"
            },
            {
                "role": "user",
                "content": "How many millimeters are in 3 centimeters?"
            },
            {
                "role": "assistant",
                "content": "1 cm = 10 mm, so 3 cm = 3 * 10 = 30 mm. <answer>30</answer>"
            },
            {
                "role": "user",
                "content": "How many meters are in 4 kilometers?"
            },
            {
                "role": "assistant",
                "content": "1 km = 1000 m, so 4 km = 4 * 1000 = 4000 m. <answer>4000</answer>"
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