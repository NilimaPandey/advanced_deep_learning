from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_4bit(x: torch.Tensor, group_size: int = 16):
    """
    Quantize a 1D tensor into 4-bit blocks.
    Returns:
      q4: packed int8 values of shape [blocks, group_size/2]
      norm: scaling factors of shape [blocks, 1]
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)                               # [blocks, group_size]
    normalization = x.abs().max(dim=-1, keepdim=True).values # [blocks, 1]

    x_norm = (x + normalization) / (2 * normalization + 1e-8)
    x_quant_8 = (x_norm * 15).round().to(torch.int8)         # [blocks, group_size]

    # pack 2x4-bit into one int8
    x_quant_4 = (x_quant_8[:, ::2] & 0xF) + ((x_quant_8[:, 1::2] & 0xF) << 4)
    return x_quant_4, normalization.to(torch.float16)        # [blocks, group_size/2], [blocks, 1]


class Linear4Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size
        blocks = in_features // group_size

        # Dummy safe buffers
        self.register_buffer(
            "weight_q4",
            torch.zeros(out_features, blocks, group_size // 2, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.ones(out_features, blocks, 1, dtype=torch.float16),
            persistent=False,
        )

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

        # Hook to quantize checkpoint weights on load
        self._register_load_state_dict_pre_hook(Linear4Bit._load_state_dict_pre_hook, with_module=True)

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]          # [out_features, in_features]
            del state_dict[f"{prefix}weight"]

            q4_list, norm_list = [], []
            for row in weight:
                q4, norm = block_quantize_4bit(row, self._group_size)
                q4_list.append(q4.unsqueeze(0))             # [1, blocks, group_size/2]
                norm_list.append(norm.unsqueeze(0))         # [1, blocks, 1]

            self.weight_q4 = torch.cat(q4_list, dim=0)      # [out_features, blocks, group_size/2]
            self.weight_norm = torch.cat(norm_list, dim=0)  # [out_features, blocks, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # quantized weights
        xq = self.weight_q4                                 # [out_features, blocks, group_size/2]
        norms = self.weight_norm.to(torch.float32)          # [out_features, blocks, 1]

        # unpack each int8 → two 4-bit values
        low = xq & 0xF                                      # [out, blocks, group_size/2]
        high = (xq >> 4) & 0xF                              # [out, blocks, group_size/2]
        q8 = torch.stack((low, high), dim=-1).reshape(xq.size(0), xq.size(1), self._group_size)
        # [out_features, blocks, group_size]

        # normalize and rescale
        q8 = q8.to(torch.float32) / 15.0
        W = (q8 * 2 * norms - norms).reshape(xq.size(0), -1)  # [out_features, in_features]

        return torch.nn.functional.linear(x.to(torch.float32), W, self.bias)


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
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNet4Bit:
    net = BigNet4Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
