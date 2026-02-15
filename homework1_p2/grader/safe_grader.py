"""
Simulation of the online grader's safe_grader.

The online grader replaces the base (frozen) part of LoRA/QLoRA layers with
identity in certain configurations to test that adapters can still learn.
This allows local verification before submission.

Based on log messages from official grader:
  "Loading lora with bigmodel replaced with identity layer at blocks [0, 2, 4] and layer 4"
  "Loading lora with bigmodel replaced with identity layer at blocks [6, 8, 10] and layer 0"
"""

from pathlib import Path

import torch

from .grader import init_loggers, grade_all
from . import tests  # noqa: F401

BIGNET_PTH = Path(__file__).parent.parent / "bignet.pth"

# Configurations observed from official grader logs
# Each tuple: (block_indices, layer_index) - replace base with identity in these layers
LORA_QLORA_REPLACE_CONFIGS = [
    ([0, 2, 4], 4),   # blocks 0,2,4, 3rd linear (index 4)
    ([6, 8, 10], 0),  # blocks 6,8,10, 1st linear (index 0)
]


def _replace_base_with_identity(model: torch.nn.Module, block_indices: list[int], layer_idx: int) -> None:
    """Replace base forward with identity in specified LoRA/QLoRA layers."""
    # model.model = Sequential(Block0, LN, Block1, LN, Block2, LN, Block3, LN, Block4, LN, Block5)
    # Block indices 0,2,4,6,8,10 map to model.model[0], [2], [4], [6], [8], [10]
    for block_idx in block_indices:
        block = model.model[block_idx]
        layer = block.model[layer_idx]
        if hasattr(layer, "lora_a"):  # LoRALinear or QLoRALinear
            def _patched_forward(x, _layer=layer):
                # Use identity instead of base: output = x + lora_out
                lora_out = _layer.lora_b(_layer.lora_a(x.to(torch.float32)))
                if hasattr(_layer, "lora_scale"):
                    lora_out = lora_out * _layer.lora_scale
                return x + lora_out.to(x.dtype)

            layer.forward = _patched_forward


def _run_accuracy_all_configs(grader, model_name: str, acc_range) -> float:
    """Run fit with each identity-replacement config, return minimum normalized accuracy."""
    import numpy as np

    min_acc = 1.0
    for block_indices, layer_idx in LORA_QLORA_REPLACE_CONFIGS:
        model = grader.load_model(model_name)
        _replace_base_with_identity(model, block_indices, layer_idx)
        acc = tests.fit_binary_classifier(model.to(grader.device), grader.TRAIN_STEPS)
        min_acc = min(min_acc, acc.item() if hasattr(acc, "item") else float(acc))
    return np.clip((min_acc - acc_range[0]) / (acc_range[1] - acc_range[0]), 0.0, 1.0)


def _patched_accuracy(self, model, min_accuracy=0.5, max_accuracy=1.0):
    """Patched accuracy that runs with identity replacement (for LoRA/QLoRA)."""
    return _run_accuracy_all_configs(self, self.KIND, (min_accuracy, max_accuracy))


def run():
    """Run grader with safe_grader (identity replacement) simulation."""
    import sys

    # Patch accuracy on LoRA and QLoRA graders to use identity replacement
    tests.LoraGrader.accuracy = _patched_accuracy
    tests.QLORAGrader.accuracy = _patched_accuracy

    logger = init_loggers(None, show_debug=False, disable_color=False)
    print("Testing grader loaded (identity replacement simulation).")
    print("Loading assignment")
    assignment = __import__("grader.grader", fromlist=["load_assignment"]).load_assignment(logger, "homework")
    if assignment is None:
        sys.exit(1)
    print("Loading grader")
    total_score = grade_all(assignment, logger, verbose=True)
    return total_score
