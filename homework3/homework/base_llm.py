from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Convert question into a prompt for the model.
        By default this is just the question (bare prompt).
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse <answer>...</answer> tag.
        If missing, fallback to first number.
        """
        import re

        m = re.search(r"<answer>(.*?)</answer>", answer)
        if m:
            text = m.group(1)
        else:
            text = answer

        num = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not num:
            return float("nan")
        try:
            return float(num.group(0))
        except:
            return float("nan")

    @overload
    def generate(self, prompt: str, max_new_tokens=50, temperature=0):
        ...

    @overload
    def generate(
        self,
        prompt: str,
        max_new_tokens=50,
        temperature=0,
        num_return_sequences: int = None,
    ):
        ...

    def generate(
        self,
        prompt: str,
        max_new_tokens=50,
        temperature=0,
        num_return_sequences: int | None = None,
    ):
        formatted = self.format_prompt(prompt)
        enc = self.tokenizer(
            formatted,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        do_sample = temperature > 0
        nrs = 1 if num_return_sequences is None else num_return_sequences

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_return_sequences=nrs,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return decoded if nrs > 1 else decoded[0]

    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: int | None = None,
        temperature: float = 0,
    ) -> list[str] | list[list[str]]:
        """
        BATCHED GENERATION IMPLEMENTATION REQUIRED BY GRADER
        """
        # IMPORTANT for decoder-only models:
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Step 1: Format prompts same as generate()
        formatted = [self.format_prompt(p) for p in prompts]

        # Step 2: Tokenize with padding
        enc = self.tokenizer(formatted, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        # Step 3: Generation settings
        do_sample = temperature > 0
        nrs = 1 if num_return_sequences is None else num_return_sequences

        out = self.model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=50,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_return_sequences=nrs,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        # Flat list if num_return_sequences is None
        if num_return_sequences is None:
            return decoded

        # Otherwise group per input
        grouped = []
        for i in range(len(prompts)):
            start = i * nrs
            grouped.append(decoded[start : start + nrs])
        return grouped

    def answer(self, *questions) -> list[float]:
        outputs = self.batched_generate(list(questions))
        return [self.parse_answer(o) for o in outputs]


def test_model():
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
