from .base_llm import BaseLLM
from .sft import SFTModel, test_model


class RFTModel(SFTModel):
    """Same prompt format as SFT/RFT training: question + trailing space."""


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = RFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def _rft_format(question: str, correct_answer: float, reasoning: str) -> dict[str, str]:
    return {"question": question, "answer": reasoning}


def train_model(
    output_dir: str,
    **kwargs,
):
    import json
    from pathlib import Path

    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    from .base_llm import BaseLLM
    from .sft import TokenizedDataset

    data_dir = Path(__file__).parent.parent / "data"
    rft_path = data_dir / "rft.json"
    with open(rft_path) as f:
        rft_data = json.load(f)

    llm = BaseLLM()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenized_dataset = TokenizedDataset(llm.tokenizer, rft_data, _rft_format)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        gradient_checkpointing=True,
        learning_rate=1e-4,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        save_strategy="no",
    )
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
