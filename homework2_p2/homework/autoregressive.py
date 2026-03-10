import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        self.embed = torch.nn.Embedding(n_tokens, d_latent)
        self.start_embed = torch.nn.Parameter(torch.randn(1, 1, d_latent) * 0.02)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=512,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.proj = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, h, w = x.shape
        L = h * w
        x_flat = x.reshape(B, L)  # (B, L)
        embedded = self.embed(x_flat)  # (B, L, d)
        # Shift: [start, x[0], x[1], ..., x[L-2]] to predict [x[0], ..., x[L-1]]
        if L > 1:
            shifted = torch.cat(
                [self.start_embed.expand(B, 1, self.d_latent), embedded[:, :-1, :]],
                dim=1,
            )
        else:
            shifted = self.start_embed.expand(B, 1, self.d_latent)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        out = self.transformer(shifted, mask=mask)  # (B, L, d)
        logits = self.proj(out)  # (B, L, n_tokens)
        return logits.reshape(B, h, w, self.n_tokens), {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        if device is None:
            device = next(self.parameters()).device
        L = h * w
        tokens = torch.zeros(B, h, w, dtype=torch.long, device=device)
        for i in range(L):
            logits, _ = self.forward(tokens)
            logits_i = logits.reshape(B, -1, self.n_tokens)[:, i, :]
            probs = torch.nn.functional.softmax(logits_i, dim=-1)
            next_tok = torch.multinomial(probs + 1e-8, 1).squeeze(-1)
            tokens.reshape(B, -1)[:, i] = next_tok
        return tokens
