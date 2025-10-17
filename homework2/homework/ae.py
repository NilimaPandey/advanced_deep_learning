import abc
import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an arbitrary tensor from (H, W, C) to (C, H, W) format.
    Works with or without a batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    """
    The opposite of hwc_to_chw.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)


class PatchifyLinear(torch.nn.Module):
    """Linear patch embedding via Conv2d."""

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.patch_conv = torch.nn.Conv2d(3, latent_dim, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return chw_to_hwc(self.patch_conv(hwc_to_chw(x)))


class UnpatchifyLinear(torch.nn.Module):
    """Linear unpatchify via ConvTranspose2d."""

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.unpatch_conv = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return chw_to_hwc(self.unpatch_conv(hwc_to_chw(x)))


class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, H, W, 3) → (B, h, w, bottleneck)."""

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode (B, h, w, bottleneck) → (B, H, W, 3)."""


class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    """Patch-level AutoEncoder."""

    class PatchEncoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.patch_conv = torch.nn.Conv2d(
                in_channels=3, out_channels=latent_dim, kernel_size=patch_size, stride=patch_size, bias=True
            )
            self.act = torch.nn.GELU()
            self.mix1 = torch.nn.Conv2d(latent_dim, latent_dim, kernel_size=1, bias=True)
            self.mix2 = torch.nn.Conv2d(latent_dim, bottleneck, kernel_size=1, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = hwc_to_chw(x)
            x = self.patch_conv(x)
            x = self.act(x)
            x = self.mix1(x)
            x = self.act(x)
            x = self.mix2(x)
            return chw_to_hwc(x)

    class PatchDecoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.mix1 = torch.nn.Conv2d(bottleneck, latent_dim, kernel_size=1, bias=True)
            self.act = torch.nn.GELU()
            self.unpatch = torch.nn.ConvTranspose2d(
                in_channels=latent_dim, out_channels=3, kernel_size=patch_size, stride=patch_size, bias=True
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = hwc_to_chw(x)
            x = self.mix1(x)
            x = self.act(x)
            x = self.unpatch(x)
            return chw_to_hwc(x)

    def __init__(self, patch_size: int = 25, latent_dim: int = 128, bottleneck: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.bottleneck = bottleneck
        self.encoder = self.PatchEncoder(patch_size, latent_dim, bottleneck)
        self.decoder = self.PatchDecoder(patch_size, latent_dim, bottleneck)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return reconstructed image and optional losses (none here)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
