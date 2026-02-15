from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_3bit(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    3-bit quantization: 8 levels (0-7), packs 8 values in 3 bytes.
    Uses less memory than 4-bit while retaining better accuracy than 2-bit.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0
    assert group_size % 8 == 0  # Pack 8 values per 3 bytes

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    normalization = torch.clamp(normalization, min=1e-8)
    x_norm = (x + normalization) / (2 * normalization)
    x_quant_8 = (x_norm * 7).round().clamp(0, 7).to(torch.int8)

    # Pack 8 consecutive 3-bit values into 3 bytes
    # Layout: v0,v1,v2,v3,v4,v5,v6,v7 -> b0,b1,b2
    # b0 = v0(3) | v1(3) | v2(2); b1 = v2(1) | v3(3) | v4(3) | v5(1); b2 = v5(2) | v6(3) | v7(3)
    n_groups = x_quant_8.size(0)
    n_packs = group_size // 8
    packed = x_quant_8.new_empty(n_groups, n_packs * 3, dtype=torch.int8)
    for i in range(n_packs):
        v = x_quant_8[:, i * 8 : (i + 1) * 8]
        b0 = v[:, 0] | ((v[:, 1] & 0x7) << 3) | ((v[:, 2] & 0x3) << 6)
        b1 = (v[:, 2] >> 2) | ((v[:, 3] & 0x7) << 1) | ((v[:, 4] & 0x7) << 4) | ((v[:, 5] & 0x1) << 7)
        b2 = (v[:, 5] >> 1) | ((v[:, 6] & 0x7) << 2) | ((v[:, 7] & 0x7) << 5)
        packed[:, i * 3] = b0
        packed[:, i * 3 + 1] = b1
        packed[:, i * 3 + 2] = b2

    return packed, normalization.to(torch.float16)


def block_dequantize_3bit(x_quant_3: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    """Reverse of block_quantize_3bit."""
    assert x_quant_3.dim() == 2
    n_packs = x_quant_3.shape[1] // 3
    group_size = n_packs * 8

    normalization = normalization.to(torch.float32)
    x_quant_8 = x_quant_3.new_empty(x_quant_3.size(0), group_size)
    for i in range(n_packs):
        b0 = x_quant_3[:, i * 3]
        b1 = x_quant_3[:, i * 3 + 1]
        b2 = x_quant_3[:, i * 3 + 2]
        x_quant_8[:, i * 8] = b0 & 0x7
        x_quant_8[:, i * 8 + 1] = (b0 >> 3) & 0x7
        x_quant_8[:, i * 8 + 2] = ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2)
        x_quant_8[:, i * 8 + 3] = (b1 >> 1) & 0x7
        x_quant_8[:, i * 8 + 4] = (b1 >> 4) & 0x7
        x_quant_8[:, i * 8 + 5] = ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1)
        x_quant_8[:, i * 8 + 6] = (b2 >> 2) & 0x7
        x_quant_8[:, i * 8 + 7] = (b2 >> 5) & 0x7

    x_norm = x_quant_8.to(torch.float32) / 7
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class Linear3Bit(torch.nn.Module):
    """Linear layer with 3-bit quantized weights. ~8x smaller than float32."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32) -> None:
        super().__init__()
        assert group_size % 8 == 0
        self._shape = (out_features, in_features)
        self._group_size = group_size
        self._n_packs = group_size // 8

        self.register_buffer(
            "weight_q3",
            torch.zeros(out_features * in_features // group_size, self._n_packs * 3, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )
        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            weight_flat = weight.flatten()
            weight_q3, weight_norm = block_quantize_3bit(weight_flat, self._group_size)
            self.weight_q3.copy_(weight_q3.view_as(self.weight_q3))
            self.weight_norm.copy_(weight_norm.view_as(self.weight_norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight = block_dequantize_3bit(self.weight_q3, self.weight_norm)
            weight = weight.view(self._shape)
            return torch.nn.functional.linear(x, weight, self.bias)


class BigNet3Bit(torch.nn.Module):
    """
    BigNet with 3-bit quantized linear layers.
    Target: <9MB memory while retaining decent accuracy.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int, group_size: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels, group_size=group_size),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=group_size),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, group_size=group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, group_size: int = 32):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNet3Bit:
    net = BigNet3Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
