from .base_llm import BaseLLM
from .data import Dataset, benchmark


def train_model(
    output_dir: str,
    **kwargs,
):
    """
    Minimal Rejection Fine-Tuning train_model.
    Must load rft.json and train a LoRA adapter.
    """
    import json
    from pathlib import Path
    from transformers import TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model

    base = BaseLLM()
    tokenizer = base.tokenizer
    model = base.model

    # LoRA config
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        bias="none",
    )

    model = get_peft_model(model, config)

    # Load offline data
    data_path = Path(__file__).parent.parent / "data" / "rft.json"
    with open(data_path) as f:
        data = json.load(f)

    class RFTDataset:
        def __len__(self): return len(data)

        def __getitem__(self, idx):
            q, y, text = data[idx]
            enc = tokenizer(text, truncation=True, padding="max_length", max_length=128)
            enc["labels"] = enc["input_ids"]
            return enc

    train_data = RFTDataset()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=5e-4,
        save_strategy="no",
        report_to="none",
        logging_steps=20,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_data)
    trainer.train()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    print(f"Saved RFT model to {out}")

    test_model(output_dir)


def test_model(ckpt_path: str):
    from peft import PeftModel
    from pathlib import Path

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, Path(ckpt_path)).to(llm.device)

    result = benchmark(llm, Dataset("valid"), 100)
    print(result.accuracy, result.answer_rate)
