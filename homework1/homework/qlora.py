from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm
from .low_precision import Linear4Bit, block_dequantize_4bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)

        # Freeze base quantized weights
        self.requires_grad_(False)

        # LoRA adapters (trainable, float32)
        self.lora_A = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_B = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)

        # Scaling factor
        self.scaling = 1.0 / lora_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward (dequantize + matmul)
        with torch.no_grad():
            out_features, in_features = self._shape
            W = []
            for i in range(out_features):
                start = i * (in_features // self._group_size)
                end = (i + 1) * (in_features // self._group_size)
                row_q4 = self.weight_q4[start:end]
                row_norm = self.weight_norm[start:end]
                row = block_dequantize_4bit(row_q4, row_norm)
                W.append(row)
            W = torch.stack(W, dim=0)
            base_out = torch.nn.functional.linear(x.to(torch.float32), W, self.bias)

        # LoRA adapters (trainable)
        lora_out = self.lora_B(self.lora_A(x.to(torch.float32))) * self.scaling

        return (base_out + lora_out).to(x.dtype)


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
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
