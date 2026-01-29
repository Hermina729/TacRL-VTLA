from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp


@dataclass
class DualFSRMLPConfig:
    # Input settings
    T: int = 5
    H: int = 16
    W: int = 16
    use_delta: bool = True

    # Encoder sizes
    hidden: int = 1024
    emb_dim: int = 512

    # Transformer token width (Gemma hidden size)
    width: int = 2048

    # Regularization
    dropout: float = 0.10


class SharedFSRMLPEncoder(nn.Module):
    """Shared encoder used for both left and right FSR windows."""

    cfg: DualFSRMLPConfig

    @nn.compact
    def __call__(self, fsr_window: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        """
        Args:
            fsr_window: (B, T, H, W)
        Returns:
            (B, emb_dim)
        """
        x = fsr_window.astype(jnp.float32)

        if self.cfg.use_delta:
            if x.shape[1] < 2:
                raise ValueError("FSR window must have T>=2 when use_delta=True.")
            delta = (x[:, -1] - x[:, -2])[:, None, :, :]
            x = jnp.concatenate([x, delta], axis=1)

        b = x.shape[0]
        x = x.reshape((b, -1))

        x = nn.Dense(self.cfg.hidden)(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dropout(rate=self.cfg.dropout)(x, deterministic=not train)

        x = nn.Dense(self.cfg.emb_dim)(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        return x


class DualFSRToTwoTokens(nn.Module):
    """Dual FSR tactile module that outputs two tokens (left/right)."""

    cfg: DualFSRMLPConfig

    @nn.compact
    def __call__(self, fsr_left: jnp.ndarray, fsr_right: jnp.ndarray, *, train: bool):
        encoder = SharedFSRMLPEncoder(self.cfg)
        emb_left = encoder(fsr_left, train=train)
        emb_right = encoder(fsr_right, train=train)

        def project(emb: jnp.ndarray) -> jnp.ndarray:
            tok = nn.Dense(self.cfg.width)(emb)
            tok = nn.LayerNorm()(tok)
            return tok[:, None, :]

        return project(emb_left), project(emb_right)
