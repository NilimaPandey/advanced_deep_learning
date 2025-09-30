from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm
from .low_precision import Linear4Bit


class QLoRALinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, lora_dim: int, group_size: int = 16, bias: bool = True):
        super().__init__()
        self._shape = (out_features, in_features)
        self.base = Linear4Bit(in_features, out_features, bias, group_size)

        # LoRA adapters - these ARE trainable
        self.lora_A = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_B = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)

        # Proper LoRA initialization
        torch.nn.init.normal_(self.lora_A.weight, std=1e-4)
        torch.nn.init.zeros_(self.lora_B.weight)

        self._register_load_state_dict_pre_hook(QLoRALinear._load_state_dict_pre_hook, with_module=True)

    @staticmethod
    def _load_state_dict_pre_hook(module, state_dict, prefix, *args, **kwargs):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            module.base._load_state_dict_pre_hook({f"{prefix}base.weight": weight}, f"{prefix}base.", {}, True, [], [],
                                                  [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output (no gradients) - keep in float32
        base_out = self.base(x)
        # LoRA output (has gradients)
        lora_out = self.lora_B(self.lora_A(x.to(torch.float32)))
        return base_out + lora_out


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