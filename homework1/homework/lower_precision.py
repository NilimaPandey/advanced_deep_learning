from pathlib import Path
import torch


def block_quantize_4bit(x: torch.Tensor, group_size: int = 32):
    """
    Quantize into 4-bit with larger group size (default 32) to save memory.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)                               # [blocks, group_size]
    normalization = x.abs().max(dim=-1, keepdim=True).values # [blocks, 1]

    x_norm = (x + normalization) / (2 * normalization + 1e-8)
    x_quant_8 = (x_norm * 15).round().to(torch.int8)         # [blocks, group_size]

    # pack 2 values into 1 byte
    x_quant_4 = (x_quant_8[:, ::2] & 0xF) + ((x_quant_8[:, 1::2] & 0xF) << 4)
    return x_quant_4, normalization.to(torch.float16)


def block_dequantize_4bit(q4: torch.Tensor, norm: torch.Tensor, group_size: int = 32):
    """
    Dequantize from packed 4-bit with scaling.
    """
    q8 = torch.empty(q4.size(0), q4.size(1) * 2, dtype=torch.int8, device=q4.device)
    q8[:, ::2] = q4 & 0xF
    q8[:, 1::2] = (q4 >> 4) & 0xF

    q8 = q8.to(torch.float32) / 15.0
    norm = norm.to(torch.float32)
    x = (q8 * 2 * norm) - norm
    return x.view(-1, group_size).reshape(-1)


class LinearGrouped4Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32):
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        blocks = in_features // group_size
        self.register_buffer("weight_q4", torch.zeros(out_features, blocks, group_size // 2, dtype=torch.int8), persistent=False)
        self.register_buffer("weight_norm", torch.ones(out_features, blocks, 1, dtype=torch.float16), persistent=False)

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

        self._register_load_state_dict_pre_hook(LinearGrouped4Bit._load_state_dict_pre_hook, with_module=True)

    def _load_state_dict_pre_hook(self, state_dict, prefix, *args, **kwargs):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]

            q4_list, norm_list = [], []
            for row in weight:
                q4, norm = block_quantize_4bit(row, self._group_size)
                q4_list.append(q4.unsqueeze(0))
                norm_list.append(norm.unsqueeze(0))

            self.weight_q4 = torch.cat(q4_list, dim=0)
            self.weight_norm = torch.cat(norm_list, dim=0)

    def forward(self, x: torch.Tensor):
        out_features, in_features = self._shape
        blocks = in_features // self._group_size

        # unpack
        q8 = torch.empty(self.weight_q4.size(0), self.weight_q4.size(1) * 2, dtype=torch.int8, device=self.weight_q4.device)
        q8[:, ::2] = self.weight_q4 & 0xF
        q8[:, 1::2] = (self.weight_q4 >> 4) & 0xF

        q8 = q8.to(torch.float32) / 15.0
        norms = self.weight_norm.to(torch.float32).expand(-1, -1, self._group_size).reshape(self.weight_q4.size(0), -1)

        W = (q8 * 2 * norms - norms)  # [out_features, in_features]
        return torch.nn.functional.linear(x.to(torch.float32), W, self.bias)


class BigNetGrouped4Bit(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, group_size=32):
            super().__init__()
            self.model = torch.nn.Sequential(
                LinearGrouped4Bit(channels, channels, group_size=group_size),
                torch.nn.ReLU(),
                LinearGrouped4Bit(channels, channels, group_size=group_size),
                torch.nn.ReLU(),
                LinearGrouped4Bit(channels, channels, group_size=group_size),
            )

        def forward(self, x):
            return self.model(x) + x

    def __init__(self, group_size=32):
        super().__init__()
        from .bignet import BIGNET_DIM, LayerNorm
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, group_size), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size),
        )

    def forward(self, x):
        return self.model(x)


def load(path: Path | None) -> BigNetGrouped4Bit:
    net = BigNetGrouped4Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
