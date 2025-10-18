import abc
import torch


def load() -> torch.nn.Module:
    from pathlib import Path
    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """Base class for all autoregressive models."""

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """(B,h,w) → (B,h,w,n_token) predicted logits."""

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        """Stub for interface; overridden below."""
        raise NotImplementedError()


# ---------------------------------------------------------------------
# Decoder-only Transformer autoregressive model
# ---------------------------------------------------------------------
class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Simple autoregressive image model.
    Input: integer tokens (B,h,w)
    Output: logits over tokens (B,h,w,n_tokens)
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens

        # --- Embedding and positional encodings ---
        self.token_embed = torch.nn.Embedding(n_tokens, d_latent)
        self.pos_embed = None  # created dynamically at runtime

        # --- Transformer encoder layers (decoder-style with causal mask) ---
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=d_latent * 4,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)

        # --- Output projection ---
        self.to_logits = torch.nn.Linear(d_latent, n_tokens)

    # --------------------------------------------------------------
    # Forward: next-token prediction
    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        x: (B,h,w) integers
        Return: logits (B,h,w,n_tokens)
        """
        B, h, w = x.shape
        seq_len = h * w

        # Flatten to sequence
        tokens = x.view(B, seq_len)

        # Token embeddings
        emb = self.token_embed(tokens)  # (B,seq,d_latent)

        # Positional embeddings
        if self.pos_embed is None or self.pos_embed.size(0) < seq_len:
            self.pos_embed = torch.nn.Parameter(
                torch.randn(seq_len, self.d_latent, device=emb.device) * 0.01,
                requires_grad=True,
            )
        else:
            self.pos_embed = torch.nn.Parameter(self.pos_embed.to(emb.device), requires_grad=True)

        emb = emb + self.pos_embed[:seq_len]

        # Shift input right by one position for autoregressive target alignment
        shifted = torch.nn.functional.pad(emb[:, :-1, :], (0, 0, 1, 0))  # prepend zero vector

        # Causal mask: no token can attend to future positions
        mask = torch.triu(torch.ones(seq_len, seq_len, device=emb.device), diagonal=1).bool()

        # Run transformer
        out = self.transformer(shifted, mask)

        # Project to token logits
        logits = self.to_logits(out)  # (B,seq,n_tokens)
        logits = logits.view(B, h, w, self.n_tokens)

        return logits, {}

    # --------------------------------------------------------------
    # Generation: sample pixels left-to-right, top-to-bottom
    # --------------------------------------------------------------
    @torch.no_grad()
    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:
        """
        Autoregressive generation loop:
        start from zeros → fill sequentially using model predictions.
        """
        if device is None:
            device = next(self.parameters()).device

        seq_len = h * w
        tokens = torch.zeros((B, seq_len), dtype=torch.long, device=device)

        for i in range(seq_len):
            emb = self.token_embed(tokens)
            if self.pos_embed is None or self.pos_embed.size(0) < seq_len:
                self.pos_embed = torch.nn.Parameter(
                    torch.randn(seq_len, self.d_latent, device=emb.device) * 0.01,
                    requires_grad=True,
                )
            else:
                self.pos_embed = torch.nn.Parameter(self.pos_embed.to(emb.device), requires_grad=True)

            emb = emb + self.pos_embed[:seq_len]

            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            out = self.transformer(emb, mask)
            logits = self.to_logits(out)  # (B,seq,n_tokens)

            # Sample next token
            probs = torch.softmax(logits[:, i, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens[:, i] = next_token.squeeze(1)

        return tokens.view(B, h, w)
