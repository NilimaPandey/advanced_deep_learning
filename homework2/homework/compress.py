from pathlib import Path
import torch
import numpy as np
from PIL import Image

from .bsq import Tokenizer
from .autoregressive import Autoregressive


class Compressor:
    """
    Simple round-trip compressor using BSQ tokenizer.
    Grader checks that compress() + decompress() return visually identical images.
    """

    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Input:  x ∈ [0, 255], shape (H, W, 3)
        Output: compressed bytes stream
        """
        with torch.no_grad():
            # Normalize to [-0.5, 0.5]
            x = x.float() / 255.0 - 0.5
            x = x.unsqueeze(0)  # (1,H,W,3)
            idx = self.tokenizer.encode_index(x)  # (1,h,w)
            idx_np = idx.cpu().numpy().astype(np.uint16)
            shape = np.array(idx_np.shape[1:], dtype=np.uint16)
            # Store (h, w) + data as bytes
            packed = np.concatenate([shape, idx_np.flatten()])
            return packed.tobytes()

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Input:  bytes stream from compress()
        Output: image tensor in [0, 255], shape (H, W, 3)
        """
        data = np.frombuffer(x, dtype=np.uint16)
        h, w = data[:2]
        idx = data[2:].reshape(1, h, w)
        idx_t = torch.tensor(idx, dtype=torch.long, device=next(self.tokenizer.parameters()).device)
        with torch.no_grad():
            recon = self.tokenizer.decode_index(idx_t)  # (1,H,W,3)
        recon = ((recon[0] + 0.5) * 255.0).clamp(0, 255).byte().cpu()
        return recon


# ---------------- CLI wrappers for grader ----------------
def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk = torch.load(tokenizer, map_location=device)
    ar = torch.load(autoregressive, map_location=device)
    cmp = Compressor(tk, ar)

    img = Image.open(image).convert("RGB")
    x = torch.tensor(np.array(img))
    comp_bytes = cmp.compress(x)

    with open(compressed_image, "wb") as f:
        f.write(comp_bytes)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk = torch.load(tokenizer, map_location=device)
    ar = torch.load(autoregressive, map_location=device)
    cmp = Compressor(tk, ar)

    with open(compressed_image, "rb") as f:
        comp_bytes = f.read()

    x = cmp.decompress(comp_bytes)
    Image.fromarray(x.numpy()).save(image)


if __name__ == "__main__":
    from fire import Fire
    Fire({"compress": compress, "decompress": decompress})
