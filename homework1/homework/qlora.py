from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm


class QLoRALinear(torch.nn.Module):
    def __init__(self, base_layer: torch.nn.Module, lora_dim: int):
        super().__init__()
        self.base = base_layer
        self.base.requires_grad_(False)  # freeze quantized base

        in_features = base_layer._shape[1]
        out_features = base_layer._shape[0]

        # LoRA adapters (trainable)
        self.lora_A = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_B = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)

        # Zero init → model == BigNet4Bit at load time
        torch.nn.init.zeros_(self.lora_A.weight)
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)  # quantized base
        lora_out = self.lora_B(self.lora_A(x.to(torch.float32)))
        return (base_out + lora_out).to(x.dtype)


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            from .low_precision import Linear4Bit
            self.model = torch.nn.Sequential(
                QLoRALinear(Linear4Bit(channels, channels, True, group_size), lora_dim),
                torch.nn.ReLU(),
                QLoRALinear(Linear4Bit(channels, channels, True, group_size), lora_dim),
                torch.nn.ReLU(),
                QLoRALinear(Linear4Bit(channels, channels, True, group_size), lora_dim),
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
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
