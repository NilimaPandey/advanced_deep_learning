from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import TrainingArguments, Trainer
from .base_llm import BaseLLM
from .data import Dataset, benchmark
import json


class RFTDataset:
    def __init__(self, tokenizer):
        data_path = Path(__file__).parent.parent / "data" / "rft.json"
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        _, _, reasoning = self.data[idx]
        enc = self.tokenizer(reasoning, truncation=True, padding="max_length", max_length=128)
        enc["labels"] = enc["input_ids"].copy()
        return enc


def train_model(output_dir="homework/rft_model"):
    base = BaseLLM()
    model = base.model
    tokenizer = base.tokenizer
    cfg = LoraConfig(
        r=16, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear"
    )
    model = get_peft_model(model, cfg)
    if torch.cuda.is_available():
        model.enable_input_require_grads()
    train_ds = RFTDataset(tokenizer)
    args = TrainingArguments(
        output_dir=output_dir, learning_rate=1e-4, num_train_epochs=3,
        per_device_train_batch_size=16, gradient_checkpointing=True,
        logging_dir=output_dir, report_to="tensorboard"
    )
    Trainer(model=model, args=args, train_dataset=train_ds).train()
    model.save_pretrained(output_dir)
    print(f"Saved -> {output_dir}")


def load():
    llm = BaseLLM()
    path = Path(__file__).parent / "rft_model"
    llm.model = PeftModel.from_pretrained(llm.model, path).to(llm.device)
    llm.model.eval()
    return llm


def test_model():
    llm = load()
    testset = Dataset("valid")
    res = benchmark(llm, testset, 100)
    print(res.accuracy, res.answer_rate)
