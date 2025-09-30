from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm


class HalfLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)

        # Cast parameters to float16
        self.weight.data = self.weight.data.half()
        if self.bias is not None:
            self.bias.data = self.bias.data.half()

        self.requires_grad_(False)  # no backprop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        out_half = torch.nn.functional.linear(x.to(torch.float16), self.weight, self.bias)
        return out_half.to(x_dtype)


class HalfBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> HalfBigNet:
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
