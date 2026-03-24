from .base_llm import BaseLLM
from .data import Dataset, benchmark


class SFTModel(BaseLLM):
    """BaseLLM with SFT adapter; uses trailing space on prompt to match training."""

    def format_prompt(self, question: str) -> str:
        return f"{question} "


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = SFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    # Merged weights often generate more reliably than active LoRA adapters at inference
    if hasattr(llm.model, "merge_and_unload"):
        llm.model = llm.model.merge_and_unload()

    return llm


# Long enough for long questions + CoT + <answer>...</answer>; avoid truncating away the answer
MAX_SEQ_LEN = 512


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    # Use "question " (with space) as prompt so boundary matches inference
    prompt_text = f"{question} "
    full_text = f"{prompt_text}{answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=MAX_SEQ_LEN)

    input_ids = full["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True, truncation=False)["input_ids"]
    # After truncation, only count prompt tokens that still appear as a prefix of input_ids
    question_len = 0
    for i, tid in enumerate(prompt_ids):
        if i >= len(input_ids) or input_ids[i] != tid:
            break
        question_len = i + 1
    # Fallback if tokenizer produced mismatch (should be rare)
    if question_len == 0:
        question_len = min(len(prompt_ids), len(input_ids))

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: float) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    ans_float = float(answer)
    # Round to reasonable precision for the LLM
    if abs(ans_float) >= 1000 or (abs(ans_float) < 0.01 and ans_float != 0):
        ans_str = f"{ans_float:.4g}"
    else:
        ans_str = f"{round(ans_float, 4)}"
    return {
        "question": prompt,
        "answer": f"<answer>{ans_str}</answer>",
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
    output_dir: str,
    **kwargs,
):
    from pathlib import Path

    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    llm = BaseLLM()
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, dataset, format_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        gradient_checkpointing=True,
        learning_rate=8e-5,
        num_train_epochs=5,
        per_device_train_batch_size=32,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = SFTModel()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()
    if hasattr(llm.model, "merge_and_unload"):
        llm.model = llm.model.merge_and_unload()

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
