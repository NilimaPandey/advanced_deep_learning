from pathlib import Path
import torch


def quantize_2bit(x: torch.Tensor):
    """
    2-bit quantization to {-1, 0, +1} with per-row scaling.
    Returns quantized weights (int8) and scale (float16).
    """
    scale = x.abs().max()
    if scale == 0:
        return torch.zeros_like(x, dtype=torch.int8), torch.tensor(1.0, dtype=torch.float16)

    # normalize and round to nearest {-1, 0, +1}
    x_scaled = x / scale
    q = torch.round(x_scaled).clamp(-1, 1).to(torch.int8)
    return q, scale.to(torch.float16)


def dequantize_2bit(q: torch.Tensor, scale: torch.Tensor):
    return q.to(torch.float32) * scale.to(torch.float32)


class Linear2Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self._shape = (out_features, in_features)

        # quantized weights and per-row scales
        self.register_buffer("weight_q", torch.zeros(out_features, in_features, dtype=torch.int8), persistent=False)
        self.register_buffer("weight_scale", torch.ones(out_features, 1, dtype=torch.float16), persistent=False)

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

        # hook for checkpoint loading
        self._register_load_state_dict_pre_hook(Linear2Bit._load_state_dict_pre_hook, with_module=True)

    def _load_state_dict_pre_hook(self, state_dict, prefix, *args, **kwargs):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]

            q_list, s_list = [], []
            for row in weight:
                q, s = quantize_2bit(row)
                q_list.append(q.unsqueeze(0))
                s_list.append(s.unsqueeze(0))

            self.weight_q = torch.cat(q_list, dim=0)      # [out_features, in_features]
            self.weight_scale = torch.cat(s_list, dim=0)  # [out_features, 1]

    def forward(self, x: torch.Tensor):
        W = dequantize_2bit(self.weight_q, self.weight_scale)
        return torch.nn.functional.linear(x.to(torch.float32), W, self.bias)


class BigNet2Bit(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear2Bit(channels, channels), torch.nn.ReLU(),
                Linear2Bit(channels, channels), torch.nn.ReLU(),
                Linear2Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        from .bignet import BIGNET_DIM, LayerNorm
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


def load(path: Path | None) -> BigNet2Bit:
    net = BigNet2Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
