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
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32,
                 share_rows: int = 32):
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size
        self._share_rows = share_rows

        blocks = in_features // group_size
        self.register_buffer("weight_q4", torch.zeros(out_features, blocks, group_size // 2, dtype=torch.int8),
                             persistent=False)

        # Reduced storage: one norm per share_rows instead of per row
        num_shared_norms = (out_features + share_rows - 1) // share_rows
        self.register_buffer("weight_norm", torch.ones(num_shared_norms, blocks, 1, dtype=torch.float16),
                             persistent=False)

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

        self._register_load_state_dict_pre_hook(LinearRowShared4Bit._load_state_dict_pre_hook, with_module=True)

    def _load_state_dict_pre_hook(self, state_dict, prefix, *args, **kwargs):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]

            all_q4 = []
            all_shared_norms = []

            # Process weight matrix in chunks of share_rows
            for row_start in range(0, weight.size(0), self._share_rows):
                row_end = min(row_start + self._share_rows, weight.size(0))
                row_chunk = weight[row_start:row_end]

                # Quantize each row with full precision
                chunk_q4 = []
                chunk_norms = []
                for row in row_chunk:
                    q4, norm = block_quantize_4bit(row, self._group_size)
                    chunk_q4.append(q4.unsqueeze(0))
                    chunk_norms.append(norm.unsqueeze(0))

                all_q4.extend(chunk_q4)

                # Average norms across this chunk for storage
                stacked_norms = torch.cat(chunk_norms, dim=0)  # [actual_rows, blocks, 1]
                shared_norm = stacked_norms.mean(dim=0, keepdim=True)  # [1, blocks, 1]
                all_shared_norms.append(shared_norm)

            self.weight_q4 = torch.cat(all_q4, dim=0)
            self.weight_norm = torch.cat(all_shared_norms, dim=0)

    def forward(self, x: torch.Tensor):
        # Move buffers to same device as input FIRST
        weight_q4 = self.weight_q4.to(x.device)
        weight_norm = self.weight_norm.to(x.device)

        # Unpack 4-bit values
        low = weight_q4 & 0xF
        high = (weight_q4 >> 4) & 0xF
        q8 = torch.stack((low, high), dim=-1).reshape(
            weight_q4.size(0),
            weight_q4.size(1),
            self._group_size
        )

        # Dequantize
        q8 = q8.to(torch.float32) / 15.0

        # Expand shared norms back to full row count
        norms = weight_norm.to(torch.float32)
        norms = norms.repeat_interleave(self._share_rows, dim=0)[:weight_q4.size(0)]
        norms = norms.expand(-1, -1, self._group_size)

        W = (q8 * 2 * norms - norms).reshape(weight_q4.size(0), -1)

        return torch.nn.functional.linear(x.to(torch.float32), W, self.bias)


class BigNetRowShared4Bit(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, group_size=32, share_rows=32):
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

    def __init__(self, group_size=32, share_rows=32):
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