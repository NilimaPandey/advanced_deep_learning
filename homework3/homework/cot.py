class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Provide a strong chat-style prompt with an example, instructing the model
        to explain the steps and return a number wrapped in <answer> tags.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that solves math and unit conversion problems. "
                    "Explain the steps and output your final numeric result using this format:\n"
                    "<answer>NUMBER</answer>"
                ),
            },
            {
                "role": "user",
                "content": "How many grams are there in 2.5 kilograms?",
            },
            {
                "role": "assistant",
                "content": "2.5 kilograms equals 2500 grams. <answer>2500</answer>",
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
