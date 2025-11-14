from .base_llm import BaseLLM


def load() -> BaseLLM:
    """
    REQUIRED BY THE GRADER.
    Load the SFT model LoRA adapter if it exists,
    otherwise return a BaseLLM instance.
    """
    from pathlib import Path
    from peft import PeftModel

    model_dir = Path(__file__).parent / "sft_model"

    llm = BaseLLM()

    # If trained adapter exists, load it
    if model_dir.exists():
        try:
            llm.model = PeftModel.from_pretrained(llm.model, model_dir).to(llm.device)
        except Exception:
            pass

    return llm
