from typing import List, Optional, Union
import re

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
    """
    Base class used by all parts of the homework.

    The grader expects:
    - The model to be SmolLM2-360M-Instruct
    - Left padding for batched generation
    - generate() and batched_generate() to use the same generation config
    - answer(*questions) to call batched_generate() and then parse numbers
    """

    def __init__(self, checkpoint: str = checkpoint):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # Left padding is required for decoder-only models in batched mode
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.model.eval()
        self.device = device

    # -------------------------------------------------------------------------
    # Prompt & parsing utilities
    # -------------------------------------------------------------------------

    def format_prompt(self, question: str) -> str:
        """
        Base prompt: just the raw question.

        CoTModel will override this to build a chat-style prompt.
        """
        return question

    def parse_answer(self, text: str) -> float:
        """
        Extract a numeric answer from model output.

        - Prefer content inside <answer>...</answer> if present
        - Otherwise fall back to the first number in the text
        - Return NaN if nothing numeric is found
        """
        # Prefer explicit <answer> tag
        m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
        if m:
            candidate = m.group(1)
        else:
            candidate = text

        num = re.search(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?", candidate)
        if not num:
            return float("nan")
        try:
            return float(num.group(0).replace(",", ""))
        except Exception:
            return float("nan")

    def _gen_kwargs(
        self,
        max_new_tokens: int,
        temperature: float,
        num_return_sequences: int,
    ) -> dict:
        """Shared kwargs for model.generate used in both single and batched modes."""
        do_sample = temperature is not None and temperature > 0.0
        kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            kwargs["temperature"] = temperature
        return kwargs

    # -------------------------------------------------------------------------
    # Unbatched generation
    # -------------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        num_return_sequences: Optional[int] = None,
    ) -> Union[str, List[str]]:
        """
        Generate a completion for a single prompt.

        This is the sequential version. It MUST behave consistently with
        batched_generate when called on a single-element batch.
        """
        text = self.format_prompt(prompt)
        enc = self.tokenizer(text, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        n = 1 if num_return_sequences is None else num_return_sequences
        gen_kwargs = self._gen_kwargs(max_new_tokens, temperature, n)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                **gen_kwargs,
            )

        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        # IMPORTANT: do NOT strip off the prompt here; the grader's reference
        # generations are based on the full decoded sequence.
        return decoded[0] if n == 1 else decoded

    # -------------------------------------------------------------------------
    # Batched generation
    # -------------------------------------------------------------------------

    def batched_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        num_return_sequences: Optional[int] = None,
    ) -> List[str]:
        """
        Generate completions for a batch of prompts.

        - Applies format_prompt() to each question
        - Uses left padding so that generation starts at the right edge
        - Returns a flat list of strings of length
          len(prompts) * num_return_sequences (if num_return_sequences is not None)
        """
        # Format prompts (CoTModel will turn these into chat templates)
        texts = [self.format_prompt(p) for p in prompts]

        enc = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        n = 1 if num_return_sequences is None else num_return_sequences
        gen_kwargs = self._gen_kwargs(max_new_tokens, temperature, n)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                **gen_kwargs,
            )

        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        # Flat list; grader will reshape if needed
        return decoded

    # -------------------------------------------------------------------------
    # Convenience: answer multiple questions at once
    # -------------------------------------------------------------------------

    def answer(self, *questions: str) -> List[float]:
        """
        Answer questions given as separate string arguments.

        NOTE: We pass the *raw* questions here; batched_generate will call
        format_prompt internally. This is crucial so CoTModel.answer() uses
        the chat template exactly once.
        """
        generations = self.batched_generate(list(questions), max_new_tokens=64, temperature=0.0)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # Simple smoke test: should not crash.
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
