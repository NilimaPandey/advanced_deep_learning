from pathlib import Path
import torch


def ternary_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().mean()
    q = torch.sign(x / (scale + 1e-8))
    q[q == -0] = 0
    return q.to(torch.int8), scale.to(torch.float16)


def ternary_dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (q.to(torch.float32) * scale.to(torch.float32))


class TernaryLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self._shape = (out_features, in_features)

        self.register_buffer("weight_q", None, persistent=False)
        self.register_buffer("weight_scale", None, persistent=False)

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

        self._register_load_state_dict_pre_hook(TernaryLinear._load_state_dict_pre_hook, with_module=True)

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]

            q_list, s_list = [], []
            for row in weight:
                q, s = ternary_quantize(row)
                q_list.append(q.unsqueeze(0))
                s_list.append(s.unsqueeze(0))

            self.weight_q = torch.cat(q_list, dim=0)
            self.weight_scale = torch.cat(s_list, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = ternary_dequantize(self.weight_q, self.weight_scale.unsqueeze(1))
        return torch.nn.functional.linear(x.to(torch.float32), W, self.bias)


class BigNetTernary(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                TernaryLinear(channels, channels),
                torch.nn.ReLU(),
                TernaryLinear(channels, channels),
                torch.nn.ReLU(),
                TernaryLinear(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # Lazy imports to avoid circular dependencies
        from .bignet import BIGNET_DIM, LayerNorm

        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNetTernary:
    net = BigNetTernary()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
