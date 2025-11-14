from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset as TorchDataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Trainer, TrainingArguments
import json

from .base_llm import BaseLLM
from .data import Dataset, benchmark
from .sft import test_model as sft_test_model  # reused in CLI


def load() -> BaseLLM:
    """Load the RFT model with its LoRA adapter."""
    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm


class RFTDataset(TorchDataset):
    """
    Dataset of reasoning trajectories produced by `datagen.generate_dataset`.

    Each row in the JSON file is [question: str, answer: float, reasoning: str].
    We train the model to reproduce the full reasoning string, which already
    contains the <answer> tag with a correct numeric answer.
    """

    def __init__(self, tokenizer, path: Optional[Path] = None, max_length: int = 256):
        if path is None:
            path = Path(__file__).parent.parent / "data" / "rft.json"
        with path.open() as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        _q, _y, reasoning = self.data[idx]
        text = reasoning
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        enc["labels"] = enc["input_ids"].copy()
        return {k: torch.tensor(v) for k, v in enc.items()}


def train_model(
    output_dir: str = "rft_model",
    num_train_epochs: int = 3,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 16,
    **kwargs,
):
    """
    Fine-tune the model on correct reasoning trajectories (RFT-style).

    This is a simple offline RL approximation: we only keep rollouts whose
    final numeric answer is correct and then train the model with supervised
    learning on those trajectories.
    """
    base = BaseLLM()
    model = base.model
    tokenizer = base.tokenizer

    config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, config)
    if base.device == "cuda":
        model.enable_input_require_grads()

    train_dataset = RFTDataset(tokenizer)

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

    save_path = Path(__file__).parent / "rft_model"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"Saved RFT LoRA adapter to {save_path}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": sft_test_model, "load": load})
