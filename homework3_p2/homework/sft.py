from .base_llm import BaseLLM
from .data import Dataset, benchmark


class SFTModel(BaseLLM):
    """BaseLLM that appends a trailing space to match what was used during training."""

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

    return llm


def tokenize(tokenizer, question: str, answer: str, max_seq_len: int = 128):
    """
    Tokenize a data element.
    We tokenize prompt and answer SEPARATELY then concatenate, avoiding BPE
    boundary issues where the tokenizer merges the space between question and
    answer into a different token depending on context.
    """
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.eos_token_id

    prompt = f"{question} "

    # Tokenize each part independently so token IDs match inference exactly
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    # Concatenate: [BOS? + prompt_tokens + answer_tokens + EOS]
    input_ids = (prompt_ids + answer_ids + [tokenizer.eos_token_id])[:max_seq_len]

    seq_len = len(input_ids)
    attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
    input_ids = input_ids + [pad_id] * (max_seq_len - seq_len)

    prompt_len = min(len(prompt_ids), max_seq_len)
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    for i in range(len(labels)):
        if attention_mask[i] == 0:
            labels[i] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def format_example(prompt: str, answer: float) -> dict[str, str]:
    """
    Construct a question / answer pair. Round the answer to make it easier for the LLM.
    """
    ans_float = float(answer)
    ans_rounded = round(ans_float, 2)
    if ans_rounded == int(ans_rounded):
        ans_str = str(int(ans_rounded))
    else:
        ans_str = f"{ans_rounded}"
    return {
        "question": prompt,
        "answer": f"<answer>{ans_str}</answer>",
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data, format_fn, max_seq_len: int = 128):
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data, max_seq_len=self.max_seq_len)


def train_model(
    output_dir: str,
    **kwargs,
):
    from pathlib import Path

    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

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

    dataset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, dataset, format_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        gradient_checkpointing=True,
        learning_rate=2e-4,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        warmup_ratio=0.1,
        save_strategy="no",
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = SFTModel()

    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
