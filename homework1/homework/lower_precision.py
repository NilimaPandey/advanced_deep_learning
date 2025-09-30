from pathlib import Path
import torch


def block_quantize_4bit(x: torch.Tensor, group_size: int = 32):
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + normalization) / (2 * normalization + 1e-8)
    x_quant_8 = (x_norm * 15).round().to(torch.int8)

    x_quant_4 = (x_quant_8[:, ::2] & 0xF) + ((x_quant_8[:, 1::2] & 0xF) << 4)
    return x_quant_4, normalization.to(torch.float16)


class LinearRowShared4Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32, share_rows: int = 16):
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size
        self._share_rows = share_rows

        blocks = in_features // group_size
        self.register_buffer("weight_q4", torch.zeros(out_features, blocks, group_size // 2, dtype=torch.int8), persistent=False)

        # fewer scale rows = out_features // share_rows
        self.register_buffer("weight_norm", torch.ones(out_features // share_rows, blocks, 1, dtype=torch.float16), persistent=False)

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

        self._register_load_state_dict_pre_hook(LinearRowShared4Bit._load_state_dict_pre_hook, with_module=True)

    def _load_state_dict_pre_hook(self, state_dict, prefix, *args, **kwargs):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]

            q4_rows, norm_rows = [], []
            for row in weight:
                q4, norm = block_quantize_4bit(row, self._group_size)
                q4_rows.append(q4.unsqueeze(0))
                norm_rows.append(norm.unsqueeze(0))

            q4_tensor = torch.cat(q4_rows, dim=0)
            norm_tensor = torch.cat(norm_rows, dim=0)

            self.weight_q4 = q4_tensor

            # share norms across groups of rows
            shared = []
            for i in range(0, norm_tensor.size(0), self._share_rows):
                shared.append(norm_tensor[i:i + self._share_rows].mean(0, keepdim=True))
            self.weight_norm = torch.cat(shared, dim=0)

    def forward(self, x: torch.Tensor):
        low = self.weight_q4 & 0xF
        high = (self.weight_q4 >> 4) & 0xF
        q8 = torch.stack((low, high), dim=-1).reshape(self.weight_q4.size(0), self.weight_q4.size(1), self._group_size)

        q8 = q8.to(torch.float32) / 15.0
        # expand shared norms back to full size
        expanded_norms = self.weight_norm.repeat_interleave(self._share_rows, dim=0)
        norms = expanded_norms.to(torch.float32).expand(-1, -1, self._group_size)

        W = (q8 * 2 * norms - norms).reshape(self.weight_q4.size(0), -1).detach()
        W = W.to(x.device)

        return torch.nn.functional.linear(x.to(torch.float32), W, self.bias)


class BigNetRowShared4Bit(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, group_size=32, share_rows=16):
            super().__init__()
            self.model = torch.nn.Sequential(
                LinearRowShared4Bit(channels, channels, group_size=group_size, share_rows=share_rows),
                torch.nn.ReLU(),
                LinearRowShared4Bit(channels, channels, group_size=group_size, share_rows=share_rows),
                torch.nn.ReLU(),
                LinearRowShared4Bit(channels, channels, group_size=group_size, share_rows=share_rows),
            )

        def forward(self, x):
            return self.model(x) + x

    def __init__(self, group_size=32, share_rows=16):
        super().__init__()
        from .bignet import BIGNET_DIM, LayerNorm
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, group_size, share_rows), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size, share_rows), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size, share_rows), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size, share_rows), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size, share_rows), LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, group_size, share_rows),
        )

    def forward(self, x):
        return self.model(x)


def load(path: Path | None) -> BigNetRowShared4Bit:
    net = BigNetRowShared4Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
