from .base_llm import BaseLLM


def load() -> BaseLLM:
    """
    REQUIRED BY THE GRADER.
    Loads the RFT model LoRA adapter if it exists,
    otherwise returns a BaseLLM instance.
    """
    from pathlib import Path
    from peft import PeftModel

    model_dir = Path(__file__).parent / "rft_model"

    llm = BaseLLM()

    if model_dir.exists():
        try:
            llm.model = PeftModel.from_pretrained(llm.model, model_dir).to(llm.device)
        except Exception:
            pass

    return llm
