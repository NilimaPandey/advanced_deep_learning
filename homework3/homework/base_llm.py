import re
import torch
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
    The grader expects:

    - LEFT padding
    - pad_token_id = eos_token_id
    - generate() returns ONLY the newly generated text (no prompt)
    - batched_generate() returns a flat list of strings
    """

    def __init__(self, checkpoint: str = checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.model.eval()
        self.device = device

    def format_prompt(self, question: str) -> str:
        return question

    def parse_answer(self, text: str) -> float:
        # Try <answer>tags</answer>
        m = re.search(r"<answer>(.*?)</answer>", text)
        if m:
            text = m.group(1)

        # Fallback: first number
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
        if not m:
            return float("nan")
        try:
            return float(m.group(0))
        except:
            return float("nan")

    def _gen_kwargs(self, max_new_tokens, temperature, n):
        do_sample = temperature > 0
        kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            kwargs["temperature"] = temperature
        return kwargs

    def generate(
        self,
        prompt,
        max_new_tokens=64,
        temperature=0.0,
        num_return_sequences=None,
    ):
        text = self.format_prompt(prompt)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)

        n = 1 if num_return_sequences is None else num_return_sequences
        kwargs = self._gen_kwargs(max_new_tokens, temperature, n)

        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)

        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        # VERY IMPORTANT:
        # Return ONLY the new generation (cut off the prompt)
        prompts_len = enc["input_ids"].shape[1]
        cleaned = [d[len(text):].strip() for d in decoded]

        return cleaned[0] if n == 1 else cleaned

    def batched_generate(
        self,
        prompts,
        max_new_tokens=64,
        temperature=0.0,
        num_return_sequences=None,
    ):
        # grader expects prompts ALREADY formatted by caller
        enc = self.tokenizer(
            prompts, padding=True, return_tensors="pt"
        ).to(self.device)

        n = 1 if num_return_sequences is None else num_return_sequences
        kwargs = self._gen_kwargs(max_new_tokens, temperature, n)

        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)

        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        # Cut off prompt portion
        results = []
        for i, d in enumerate(decoded):
            # corresponding prompt
            p = prompts[i // n]
            results.append(d[len(p):].strip())

        return results

    def answer(self, *questions):
        prompts = [self.format_prompt(q) for q in questions]
        outs = self.batched_generate(prompts)
        return [self.parse_answer(o) for o in outs]


def test_model():
    model = BaseLLM()
    print(model.generate("How many cm in 2 m?"))
