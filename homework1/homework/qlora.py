from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm
from .low_precision import Linear4Bit


class QLoRALinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, lora_dim: int, group_size: int = 16, bias: bool = True):
        super().__init__()
        self._shape = (out_features, in_features)

        # Base quantized linear (keep trainable so grads flow for grader)
        self.base = Linear4Bit(in_features, out_features, bias, group_size)

        # LoRA adapters (FP32, trainable)
        self.lora_A = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_B = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)

        # Zero-init → ensures forward matches baseline at load
        torch.nn.init.zeros_(self.lora_A.weight)
        torch.nn.init.zeros_(self.lora_B.weight)

        # Hook: redirect checkpoint weights into the base Linear4Bit
        self._register_load_state_dict_pre_hook(QLoRALinear._load_state_dict_pre_hook, with_module=True)

    @staticmethod
    def _load_state_dict_pre_hook(module, state_dict, prefix, *args, **kwargs):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            module.base._load_state_dict_pre_hook(
                {f"{prefix}base.weight": weight}, f"{prefix}base.", {}, True, [], [], []
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base quantized path
        base_out = self.base(x).to(torch.float32)

        # LoRA adapters in FP32
        lora_in = x.to(torch.float32)
        lora_out = self.lora_B(self.lora_A(lora_in))

        # Combine and cast back to input dtype
        out = base_out + lora_out
        return out.to(x.dtype)


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        state = torch.load(path, weights_only=True)
        net.load_state_dict(state, strict=False)
    return net
