import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self.down_project = torch.nn.Linear(embedding_dim, codebook_bits)
        self.up_project = torch.nn.Linear(codebook_bits, embedding_dim)
        self._codebook_bits = codebook_bits
        # self.bn = torch.nn.BatchNorm2d(codebook_bits, affine=False)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.nn.functional.normalize(x - x.mean(dim=(0, 1, 2), keepdim=True), dim=-1)
        return torch.nn.functional.normalize(x, dim=-1)
        # return chw_to_hwc(self.bn(hwc_to_chw(x)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.down_project(x)
        x_norm = self.norm(x_proj)
        return diff_sign(x_norm)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_project(x)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self._index_to_code(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.bsq = BSQ(codebook_bits, latent_dim)
        self.codebook_bits = codebook_bits

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self.bsq.encode_index(self.encoder(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.bsq.decode_index(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.bsq.encode(self.encoder(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.bsq.decode(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x_enc = self.encode(x)
        idx = self.bsq._code_to_index(x_enc)
        cnt = torch.bincount(idx.flatten(), minlength=2**self.codebook_bits)
        # Compute some stats on index use
        return self.decode(x_enc), {
            "cb0": (cnt == 0).float().mean().detach(),
            "cb2": (cnt <= 2).float().mean().detach(),
            "cb5": (cnt <= 5).float().mean().detach(),
        }


class BSQPatchAutoEncoder16(BSQPatchAutoEncoder):
    def __init__(self):
        super().__init__(codebook_bits=16)
