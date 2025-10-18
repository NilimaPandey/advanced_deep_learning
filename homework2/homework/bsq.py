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
    Differentiable sign function (straight-through estimator).
    Returns -1 for negative values and +1 for non-negative ones.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """Abstract tokenizer base class."""

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """(B,H,W,3) → (B,h,w) integer tokens."""

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """(B,h,w) tokens → (B,H,W,3) image."""


# ---------------------------------------------------------------------
# Binary Spherical Quantization
# ---------------------------------------------------------------------
class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self._embedding_dim = embedding_dim
        self.down = torch.nn.Linear(embedding_dim, codebook_bits, bias=True)
        self.up = torch.nn.Linear(codebook_bits, embedding_dim, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Linear down-projection → L2-norm → differentiable binarization."""
        y = self.down(x)
        norm = torch.linalg.vector_norm(y, dim=-1, keepdim=True).clamp_min(1e-6)
        y = y / norm
        return diff_sign(y)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Linear up-projection back to latent dimension."""
        return self.up(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # --- Helpers for token index conversion ---
    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * (2 ** torch.arange(x.size(-1)).to(x.device))).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


# ---------------------------------------------------------------------
# Combined Patch AutoEncoder + Quantizer
# ---------------------------------------------------------------------
class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """PatchAutoEncoder with a BSQ quantization bottleneck."""

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim, bottleneck=latent_dim)
        self.codebook_bits = codebook_bits
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)

    # -----------------------------------------------------------------
    # Tokenizer interface
    # -----------------------------------------------------------------
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image → integer tokens."""
        z = self.encoder(x)              # (B,h,w,latent)
        code = self.bsq.encode(z)        # (B,h,w,bits)
        return self.bsq._code_to_index(code)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """Decode integer tokens → image."""
        code = self.bsq._index_to_code(x)
        z_q = self.bsq.decode(code)
        return self.decoder(z_q)

    # -----------------------------------------------------------------
    # PatchAutoEncoder overrides
    # -----------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return quantized latent embedding."""
        z = self.encoder(x)
        code = self.bsq.encode(z)
        return self.bsq.decode(code)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Full AE forward with monitoring of codebook usage."""
        z = self.encoder(x)
        code = self.bsq.encode(z)
        z_q = self.bsq.decode(code)
        x_hat = self.decoder(z_q)

        # --- Monitor codebook entropy and usage ---
        with torch.no_grad():
            idx = self.bsq._code_to_index(code)
            cnt = torch.bincount(idx.flatten(), minlength=2 ** self.codebook_bits).float()
            total = cnt.sum().clamp_min(1.0)
            p = cnt / total
            entropy = -(p[p > 0] * torch.log2(p[p > 0])).sum()
            frac_unused = (cnt == 0).float().mean()

        logs = {
            "code_entropy": entropy.detach(),
            "code_frac_unused": frac_unused.detach(),
        }
        return x_hat, logs
