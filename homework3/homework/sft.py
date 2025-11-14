from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    """Load the SFT model with LoRA adapters applied."""
    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm


class SFTDataset(TorchDataset):
    """Supervised fine-tuning dataset built from the numeric conversion data."""

    def __init__(self, tokenizer, split: str = "train", max_length: int = 128):
        self.tokenizer = tokenizer
        self.data = Dataset(split).data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        question, answer = self.data[idx]
        # Train the model to produce the numeric answer in <answer> tags
        text = f"{question}\n<answer>{answer}</answer>{self.tokenizer.eos_token}"
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        enc["labels"] = enc["input_ids"].copy()
        return {k: torch.tensor(v) for k, v in enc.items()}


def train_model(
    output_dir: str = "sft_model",
    num_train_epochs: int = 3,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 16,
    **kwargs,
):
    """Supervised fine-tuning of SmolLM2 with a LoRA adapter."""
    base = BaseLLM()
    model = base.model
    tokenizer = base.tokenizer

    # Configure LoRA
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, config)
    if base.device == "cuda":
        model.enable_input_require_grads()

    train_dataset = SFTDataset(tokenizer, split="train")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_checkpointing=True,
        logging_dir=output_dir,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    trainer.train()

    # Save only the adapter weights
    save_path = Path(__file__).parent / "sft_model"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Saved SFT LoRA adapter to {save_path}")


def test_model(ckpt_path: Optional[str] = None):
    """
    Evaluate a LoRA checkpoint on the held-out test set.

    If ckpt_path is None, defaults to the standard SFT model directory.
    This function is re-used by the RFT code.
    """
    if ckpt_path is None:
        ckpt_path = Path(__file__).parent / "sft_model"
    else:
        ckpt_path = Path(ckpt_path)

    llm = BaseLLM()

    # Load the model with LoRA adapters
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    testset = Dataset("test")
    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
