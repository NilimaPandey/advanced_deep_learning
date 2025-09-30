from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_4bit(x: torch.Tensor, group_size: int = 16):
    assert x.dim() == 1
    assert x.size(0) % group_size == 0
    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + normalization) / (2 * normalization + 1e-8)
    x_quant_8 = (x_norm * 15).round().to(torch.int8)
    x_quant_4 = (x_quant_8[:, ::2] & 0xF) + ((x_quant_8[:, 1::2] & 0xF) << 4)
    return x_quant_4, normalization.to(torch.float16)


def block_dequantize_4bit(x_quant_4: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    assert x_quant_4.dim() == 2
    normalization = normalization.to(torch.float32)
    x_quant_8 = x_quant_4.new_empty(x_quant_4.size(0), x_quant_4.shape[1] * 2)
    x_quant_8[:, ::2] = x_quant_4 & 0xF
    x_quant_8[:, 1::2] = (x_quant_4 >> 4) & 0xF
    x_norm = x_quant_8.to(torch.float32) / 15
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class Linear4Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        # Buffers for quantized and dequantized weights
        self.register_buffer("weight_q4", torch.zeros(1, 1, dtype=torch.int8), persistent=False)
        self.register_buffer("weight_norm", torch.zeros(1, 1, dtype=torch.float16), persistent=False)
        self.register_buffer("weight_fp32", torch.zeros(out_features, in_features, dtype=torch.float32), persistent=False)

        # Optional bias
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

        # Hook to quantize checkpoint weights on load
        self._register_load_state_dict_pre_hook(Linear4Bit._load_state_dict_pre_hook, with_module=True)

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]

            out_features, in_features = self._shape
            rows_q4, rows_norm, rows_deq = [], [], []
            for row in weight:
                q4, norm = block_quantize_4bit(row, self._group_size)
                rows_q4.append(q4)
                rows_norm.append(norm)
                rows_deq.append(block_dequantize_4bit(q4, norm))

            self.weight_q4 = torch.cat(rows_q4, dim=0)
            self.weight_norm = torch.cat(rows_norm, dim=0)
            self.weight_fp32 = torch.stack(rows_deq, dim=0).to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x.to(torch.float32), self.weight_fp32, self.bias)


class BigNet4Bit(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear4Bit(channels, channels),
                torch.nn.ReLU(),
                Linear4Bit(channels, channels),
                torch.nn.ReLU(),
                Linear4Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
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


def load(path: Path | None) -> BigNet4Bit:
    net = BigNet4Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
