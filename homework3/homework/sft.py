from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import TrainingArguments, Trainer
from .base_llm import BaseLLM
from .data import Dataset, benchmark


def format_example(q, y):
    return {"question": q, "answer": f"<answer>{y}</answer>"}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        q, y = self.data[idx]
        text = f"{q} <answer>{y}</answer>{self.tokenizer.eos_token}"
        enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=128)
        enc["labels"] = enc["input_ids"].copy()
        return enc


def train_model(output_dir="homework/sft_model"):
    base = BaseLLM()
    model = base.model
    tokenizer = base.tokenizer
    peft_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear"
    )
    model = get_peft_model(model, peft_config)
    if torch.cuda.is_available():
        model.enable_input_require_grads()

    train_ds = TokenizedDataset(tokenizer, Dataset("train"))
    args = TrainingArguments(
        output_dir=output_dir, learning_rate=1e-4, num_train_epochs=3,
        per_device_train_batch_size=16, gradient_checkpointing=True,
        logging_dir=output_dir, report_to="tensorboard"
    )
    Trainer(model=model, args=args, train_dataset=train_ds).train()
    model.save_pretrained(output_dir)
    print(f"Saved adapter -> {output_dir}")


def load():
    llm = BaseLLM()
    model_path = Path(__file__).parent / "sft_model"
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm


def test_model():
    llm = load()
    testset = Dataset("valid")
    res = benchmark(llm, testset, 100)
    print(res.accuracy, res.answer_rate)
