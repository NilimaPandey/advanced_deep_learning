from typing import List, Optional, Union
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint: str = checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
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
        snippet = self._extract_between(answer, "<answer>", "</answer>")
        if snippet is None:
            return float("nan")
        m = re.search(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?", snippet)
        if not m:
            return float("nan")
        try:
            return float(m.group(0).replace(",", ""))
        except Exception:
            return float("nan")

    def _build_gen_kwargs(self, max_new_tokens, temperature, num_return_sequences):
        do_sample = temperature and temperature > 0
        return dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences or 1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def generate(self, prompt: str, max_new_tokens=64, temperature=0.0, num_return_sequences=None):
        text = self.format_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        kwargs = self._build_gen_kwargs(max_new_tokens, temperature, num_return_sequences)
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return decoded if (num_return_sequences and num_return_sequences > 1) else decoded[0]

    def batched_generate(self, prompts: List[str], max_new_tokens=64, temperature=0.0, num_return_sequences=None):
        enc = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        kwargs = self._build_gen_kwargs(max_new_tokens, temperature, num_return_sequences)
        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        nrs = kwargs["num_return_sequences"]
        if nrs > 1:
            regroup = []
            for i in range(len(prompts)):
                regroup.append(decoded[i * nrs:(i + 1) * nrs])
            return regroup
        else:
            return decoded

    def answer(self, *questions: str):
        gens = self.batched_generate(list(questions))
        return [self.parse_answer(g) for g in gens]


def test_model():
    model = BaseLLM()
    for q in ["The cat went up", "The dog went down"]:
        print(model.generate(q))
    print(model.batched_generate(["1kg in grams?", "2m in cm?"]))


if __name__ == "__main__":
    test_model()
