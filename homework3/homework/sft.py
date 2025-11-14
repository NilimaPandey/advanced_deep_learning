from .base_llm import BaseLLM
from .data import Dataset, benchmark


class SFTModel(BaseLLM):
    """SFT model - uses chat template like CoT but with direct answers."""

    def format_prompt(self, question: str) -> str:
        """Use chat template for consistency."""
        messages = [
            {
                "role": "user",
                "content": question
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt


def load() -> SFTModel:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = SFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # Round answer to whole number when possible, otherwise 1 decimal
    rounded_answer = round(float(answer))
    if abs(rounded_answer - float(answer)) > 0.01:
        rounded_answer = round(float(answer), 1)

    # Use chat template format for better results
    formatted_question = prompt

    # Simple answer with just the number in tags
    formatted_answer = f"<answer>{rounded_answer}</answer>"

    return {
        "question": formatted_question,
        "answer": formatted_answer
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
        output_dir: str = "homework/sft_model",
        num_train_epochs: int = 5,
        per_device_train_batch_size: int = 32,
        learning_rate: float = 2e-4,  # Back to proven learning rate
        **kwargs,
):
    """
    Train the model using SFT (Supervised Fine-Tuning).
    """
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    # Load base model with SFT formatting
    llm = SFTModel()

    # Configure LoRA with proven settings
    lora_config = LoraConfig(
        r=12,  # Good balance between capacity and size
        lora_alpha=48,  # 4x the rank
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to the model
    llm.model = get_peft_model(llm.model, lora_config)

    # Enable input gradients for GPU (to avoid bug with gradient_checkpointing)
    if llm.device == "cuda":
        llm.model.enable_input_require_grads()

    # Print trainable parameters
    llm.model.print_trainable_parameters()

    # Load and prepare dataset with chat template
    train_data = Dataset("train")

    # Format data using chat template
    class ChatFormattedDataset:
        def __init__(self, tokenizer, data, format_fn):
            self.tokenizer = tokenizer
            self.data = data
            self.format_fn = format_fn

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            question, answer = self.data[idx]
            formatted = self.format_fn(question, answer)

            # Create chat format
            messages = [
                {"role": "user", "content": formatted["question"]}
            ]

            # Apply chat template to question
            question_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            return tokenize(self.tokenizer, question_text, formatted["answer"])

    train_dataset = ChatFormattedDataset(llm.tokenizer, train_data, format_example)

    # Set up training arguments with better settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,  # Only keep 2 checkpoints
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        warmup_steps=50,
        weight_decay=0.01,  # Add weight decay for regularization
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


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = SFTModel()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})