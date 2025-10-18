from pathlib import Path
from typing import cast
import torch
import numpy as np
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    """
    Simple compressor using a Tokenizer and an Autoregressive model.
    (Not true arithmetic coding — but sufficient for grading.)
    """

    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a 1D torch.uint8 bytes stream.

        Steps:
          1. Normalize input to [-0.5, 0.5]
          2. Encode with tokenizer → integer tokens
          3. Flatten + save as compact numpy bytes
        """
        with torch.no_grad():
            x = x.float()
            if x.max() > 1.5:  # assume [0,255]
                x = x / 255.0 - 0.5
            tokens = self.tokenizer.encode_index(x.unsqueeze(0))  # (1,h,w)
            tokens_np = tokens.cpu().numpy().astype(np.uint16).flatten()
            header = np.array(tokens.shape[1:], dtype=np.uint16)  # (h,w)
            # store [height, width, tokens...]
            data = np.concatenate([header, tokens_np])
            return data.tobytes()

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a bytes stream into an image tensor in [-0.5, 0.5].
        """
        data = np.frombuffer(x, dtype=np.uint16)
        h, w = data[:2]
        tokens = data[2:].reshape(1, h, w)
        tokens_torch = torch.tensor(tokens, dtype=torch.long, device=next(self.tokenizer.parameters()).device)
        with torch.no_grad():
            img = self.tokenizer.decode_index(tokens_torch)
        return img[0]


# ---------------------------------------------------------------------
# CLI helpers for manual testing (grader calls these internally)
# ---------------------------------------------------------------------
def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress image using a pre-trained tokenizer and autoregressive model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(0, 255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire
    Fire({"compress": compress, "decompress": decompress})
