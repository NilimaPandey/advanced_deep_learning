from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
        output_dir: str = "homework/rft_model",
        generated_data_path: str = "data/rft.json",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 32,
        learning_rate: float = 2e-4,
        **kwargs,
):
    """
    Train the model using RFT (Rejection Fine-Tuning).

    This is similar to SFT but uses the generated dataset from datagen.py
    which contains only correct reasoning chains.

    Args:
        output_dir: Directory to save the trained model
        generated_data_path: Path to the generated dataset JSON file
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size for training
        learning_rate: Learning rate
    """
    import json
    from pathlib import Path

    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    from .sft import tokenize

    # Load base model
    llm = BaseLLM()

    # Configure LoRA - can use slightly larger rank for RFT (as mentioned in README)
    lora_config = LoraConfig(
        r=12,  # Slightly larger than SFT but still under 50MB total
        lora_alpha=48,  # 4x the rank
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to the model
    llm.model = get_peft_model(llm.model, lora_config)

    # Enable input gradients for GPU
    if llm.device == "cuda":
        llm.model.enable_input_require_grads()

    # Print trainable parameters
    llm.model.print_trainable_parameters()

    # Load generated dataset
    data_path = Path(generated_data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Generated dataset not found at {generated_data_path}. "
            "Please run datagen.py first to generate the training data.\n"
            f"Example: python -m homework.datagen {generated_data_path}"
        )

    with data_path.open('r') as f:
        generated_data = json.load(f)

    print(f"Loaded {len(generated_data)} training samples from {generated_data_path}")

    # Create a custom dataset class for RFT
    # Data format from datagen: [question, answer, reasoning]
    class RFTDataset:
        def __init__(self, tokenizer, data):
            self.tokenizer = tokenizer
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            # Format: [question, answer, reasoning]
            question = item[0]
            reasoning = item[2]  # Full response including <answer> tags

            # Tokenize the question and reasoning
            return tokenize(self.tokenizer, question, reasoning)

    # Create dataset
    train_dataset = RFTDataset(llm.tokenizer, generated_data)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logging_steps=100,
        save_strategy="epoch",
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        warmup_steps=100,
        fp16=llm.device == "cuda",
        **kwargs
    )

    # Create trainer
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train
    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)

    # Test the model
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})