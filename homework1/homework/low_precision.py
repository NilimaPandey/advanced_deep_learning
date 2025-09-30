from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm
from .low_precision import Linear4Bit


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

        # Freeze quantized base
        self.requires_grad_(False)

        # LoRA adapters (trainable, float32)
        self.lora_A = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_B = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)

        # Careful init
        torch.nn.init.normal_(self.lora_A.weight, std=1e-4)
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Base forward: dequantize vectorized ---
        xq = self.weight_q4
        norms = self.weight_norm.to(torch.float32)
        out_features, in_features = self._shape
        blocks = in_features // self._group_size

        # unpack 4-bit to 8-bit
        q8 = torch.empty(xq.size(0), xq.size(1) * 2, dtype=torch.int8, device=xq.device)
        q8[:, ::2] = xq & 0xF
        q8[:, 1::2] = (xq >> 4) & 0xF

        # normalize [0,1], rescale
        q8 = q8.to(torch.float32) / 15.0
        norms = norms.expand(-1, -1, self._group_size).reshape(xq.size(0), -1)
        W = (q8 * 2 * norms) - norms

        base_out = torch.nn.functional.linear(x.to(torch.float32), W, self.bias)

        # --- LoRA adapters ---
        lora_out = self.lora_B(self.lora_A(x.to(torch.float32)))

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
