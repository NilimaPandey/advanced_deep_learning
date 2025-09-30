from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    def __init__(self, in_features: int, out_features: int, lora_dim: int = 32, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.requires_grad_(False)

        self.lora_A = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_B = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)

        torch.nn.init.normal_(self.lora_A.weight, std=1e-4)
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = super().forward(x)
        lora_out = self.lora_B(self.lora_A(x.to(torch.float32)))
        return base_out + lora_out.to(x.dtype)


class LoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoRABigNet:
    net = LoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
