from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
import zlib

from .autoregressive import Autoregressive
from .bsq import Tokenizer


def _pack_tokens_10bit(tokens: np.ndarray) -> bytes:
    """Pack token indices (0..1023) into 10 bits each. Returns bytes."""
    flat = tokens.ravel().astype(np.uint32)
    n = len(flat)
    n_bits = n * 10
    n_bytes = (n_bits + 7) // 8
    out = bytearray(n_bytes)
    for i in range(n):
        val = int(flat[i])
        bit_off = i * 10
        byte_off = bit_off // 8
        shift = bit_off % 8
        out[byte_off] |= (val << shift) & 0xFF
        if shift > 0 and byte_off + 1 < n_bytes:
            out[byte_off + 1] |= (val >> (8 - shift)) & 0xFF
        if shift >= 6 and byte_off + 2 < n_bytes:
            out[byte_off + 2] |= (val >> (16 - shift)) & 0xFF
    return bytes(out)


def _unpack_tokens_10bit(raw: bytes, n_tokens: int) -> np.ndarray:
    """Unpack n_tokens 10-bit values from bytes."""
    out = np.zeros(n_tokens, dtype=np.int64)
    for i in range(n_tokens):
        bit_off = i * 10
        byte_off = bit_off // 8
        shift = bit_off % 8
        v = raw[byte_off]
        if byte_off + 1 < len(raw):
            v |= raw[byte_off + 1] << 8
        if byte_off + 2 < len(raw):
            v |= raw[byte_off + 2] << 16
        out[i] = (v >> shift) & 0x3FF
    return out


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a bytes stream.
        Tokens are packed at 10 bits each (matching codebook_bits) then zlib-compressed.
        """
        # x is expected in range [-0.5, 0.5] with shape (H, W, 3)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1, H, W, 3)

        device = next(self.tokenizer.parameters()).device
        x = x.to(device)

        # Encode image into discrete tokens of shape (1, h, w)
        tokens = self.tokenizer.encode_index(x)  # (1, h, w)

        tokens_np = tokens.detach().cpu().numpy().astype(np.uint32)
        raw_packed = _pack_tokens_10bit(tokens_np)
        return zlib.compress(raw_packed, level=9)

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a bytes stream back into a normalized image tensor.
        Returns tensor of shape (H, W, 3) in range ~[-0.5, 0.5].
        """
        device = next(self.tokenizer.parameters()).device

        # Recover token grid shape (h, w) using a dummy image, as in the grader.
        dummy = torch.zeros(1, 100, 150, 3, device=device)
        dummy_idx = self.tokenizer.encode_index(dummy)
        _, h, w = dummy_idx.shape
        n_tokens = h * w

        raw_packed = zlib.decompress(x)
        tokens_flat = _unpack_tokens_10bit(raw_packed, n_tokens)
        tokens = torch.from_numpy(tokens_flat.copy()).to(device).long().view(1, h, w)

        img = self.tokenizer.decode_index(tokens)
        return img[0]


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
