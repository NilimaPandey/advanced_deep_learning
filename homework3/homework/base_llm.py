from typing import overload
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class BaseLLM:
    """
    Base class for LLMs used in the homework.
    """

    def __init__(self, checkpoint: str = checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # Left padding REQUIRED for decoder-only models
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device
        self.model.eval()

    def format_prompt(self, question: str) -> str:
        return question

    @staticmethod
    def _extract_between(text: str, start: str, end: str):
        try:
            i = text.index(start) + len(start)
            j = text.index(end, i)
            return text[i:j]
        except ValueError:
            return None

    def parse_answer(self, answer: str) -> float:
        """
        Parse <answer> tag, or fallback to first number.
        """
        snippet = BaseLLM._extract_between(answer, "<answer>", "</answer>")
        if snippet is None:
            m = re.search(r"[-+]?\d+(?:\.\d+)?", answer)
            if m is None:
                return float("nan")
            snippet = m.group(0)

        try:
            return float(snippet)
        except:
            return float("nan")

    def _generation_parameters(
            self,
            temperature: float,
            num_return_sequences: int,
    ) -> dict:
        do_sample = temperature > 0.0

        return dict(
            max_new_tokens=128,  # Increased for longer reasoning
            do_sample=do_sample,
            temperature=temperature if do_sample else 0.7,  # Even when not sampling, use some temp
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.05,  # Reduced penalty
            no_repeat_ngram_size=3,  # Prevent repeating 3-grams
        )

    # -----------------------------------------------------------------

    @overload
    def generate(self, prompt: str, temperature: float = 0.0): ...
    @overload
    def generate(self, prompt: str, temperature: float, num_return_sequences: int): ...

    def generate(self, prompt: str, temperature: float = 0.0, num_return_sequences: int | None = None):
        nrs = 1 if num_return_sequences is None else num_return_sequences

        formatted = self.format_prompt(prompt)
        enc = self.tokenizer(formatted, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        params = self._generation_parameters(temperature, nrs)

        with torch.no_grad():
            out = self.model.generate(**enc, **params)

        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return decoded[0] if nrs == 1 else decoded

    # -----------------------------------------------------------------

    @overload
    def batched_generate(self, prompts: list[str], temperature: float = 0.0): ...
    @overload
    def batched_generate(self, prompts: list[str], temperature: float, num_return_sequences: int): ...

    def batched_generate(self, prompts: list[str], temperature: float = 0.0, num_return_sequences: int | None = None):
        nrs = 1 if num_return_sequences is None else num_return_sequences

        formatted = [self.format_prompt(p) for p in prompts]

        batch = self.tokenizer(
            formatted,
            padding=True,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        params = self._generation_parameters(temperature, nrs)

        with torch.no_grad():
            out = self.model.generate(**batch, **params)

        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        if num_return_sequences is None:
            return decoded

        grouped = []
        for i in range(len(prompts)):
            grouped.append(decoded[i * nrs : (i + 1) * nrs])
        return grouped

    # -----------------------------------------------------------------

    def answer(self, *questions: str):
        outs = self.batched_generate(list(questions))
        return [self.parse_answer(o) for o in outs]


def test_model():
    model = BaseLLM()
    print(model.generate("1 kg in grams?"))
    print(model.batched_generate(["1 kg in grams?", "2 m in cm?"]))
