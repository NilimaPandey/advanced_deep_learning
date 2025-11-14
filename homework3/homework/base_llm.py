from typing import List, Optional, Union, overload

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint: str = checkpoint):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # IMPORTANT: left padding for decoder-only model in batched generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            # Fall back to eos token as pad if needed
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device
        self.model.eval()

    def format_prompt(self, question: str) -> str:
        """
        Convert a raw question into a model input string.

        The base model uses a plain text prompt; CoTModel will override this
        to provide a richer chat-style prompt.
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Extract a numeric answer from the model output.

        First try to look inside <answer>...</answer>.
        If that fails, fall back to the first number in the string.
        """
        # Try inside <answer>...</answer>
        try:
            start = answer.index("<answer>") + len("<answer>")
            end = answer.index("</answer>", start)
            candidate = answer[start:end]
        except ValueError:
            candidate = answer

        m = re.search(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?", candidate)
        if not m:
            return float("nan")
        try:
            return float(m.group(0).replace(",", ""))
        except Exception:
            return float("nan")

    def _generation_kwargs(
        self,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        num_return_sequences: int = 1,
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

    @overload
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        num_return_sequences: None = None,
    ) -> str: ...
    @overload
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        num_return_sequences: int,
    ) -> list[str]: ...

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        num_return_sequences: Optional[int] = None,
    ) -> Union[str, list[str]]:
        """
        Generate a completion for a single prompt.

        This is the unbatched version that the batched implementation should mimic.
        """
        text = self.format_prompt(prompt)
        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        nrs = 1 if num_return_sequences is None else num_return_sequences
        gen_kwargs = self._generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=nrs,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if nrs == 1:
            return decoded[0]
        else:
            return decoded

    def batched_generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        num_return_sequences: Optional[int] = None,
    ) -> list[str]:
        """
        Generate completions for a batch of prompts.

        We tokenize a list of prompts with padding=True so all sequences
        have the same length. Because padding_side="left", sequences align
        on the right (where generation starts). We then call model.generate
        once for the entire batch and decode with batch_decode.

        If num_return_sequences is not None, Hugging Face flattens the outputs
        into a list of length batch_size * num_return_sequences; we keep that
        flat structure.
        """
        # NOTE: BaseLLM.answer already calls format_prompt, so these are "ready"
        enc = self.tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        nrs = 1 if num_return_sequences is None else num_return_sequences
        gen_kwargs = self._generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=nrs,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.

        This uses batched generation under the hood for efficiency.
        """
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
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
