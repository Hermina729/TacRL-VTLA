from __future__ import annotations

from dataclasses import dataclass, field

import flax.linen as nn
import jax.numpy as jnp


@dataclass
class DualFSR3DCNNConfig:
    # Input settings
    T: int = 5
    H: int = 16
    W: int = 16

    # 2D CNN channel progression (spatial downsampling)
    # Each layer uses stride 2, so H,W shrink by 2^len(channels).
    # Default: 16 / 2^3 = 2  =>  2 x 2 = 4 tokens per hand, 8 total.
    channels: tuple[int, ...] = (64, 128, 256)

    tokens_per_hand: int = 4

    # Transformer token width (Gemma hidden size)
    width: int = 2048

    # Regularization
    dropout: float = 0.10


class SharedFSR2DTemporalEncoder(nn.Module):
    """Per-frame 2D CNN + temporal 1D Conv encoder for FSR tactile windows.

    Architecture:
        1. Per-frame 2D CNN (spatial downsampling):
           (B, T, H, W) → (B*T, H, W, 1) → Conv2D blocks → (B*T, h, w, C)
        2. Reshape to spatial tokens:
           → (B, T, h*w, C)
        3. Temporal 1D Conv per spatial token (2 layers, receptive field = 5):
           → (B*h*w, T, C) → Conv1D blocks → (B*h*w, T, C)
        4. Temporal aggregation (max pool over T):
           → (B, h*w, C)
        5. Project to transformer token width:
           → (B, tokens_per_hand, width)

    With defaults (T=5, H=W=16, channels=(64,128,256)):
        (B, 5, 16, 16) → 2D CNN → (B*5, 2, 2, 256)
        → reshape → (B, 5, 4, 256)
        → temporal conv → (B*4, 5, 256) → max pool → (B, 4, 256)
        → project → (B, 4, 2048)
    """

    cfg: DualFSR3DCNNConfig

    @nn.compact
    def __call__(self, fsr_window: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        """
        Args:
            fsr_window: (B, T, H, W)
        Returns:
            (B, tokens_per_hand, width)
        """
        cfg = self.cfg
        x = fsr_window.astype(jnp.float32)
        B, T, H, W = x.shape

        # --- 1) Per-frame 2D CNN ---
        x = x.reshape(B * T, H, W, 1)
        for ch in cfg.channels:
            x = nn.Conv(ch, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.gelu(x)
            x = nn.GroupNorm(num_groups=min(8, ch))(x)
        # (B*T, h, w, C)
        _, h, w, C = x.shape

        # --- 2) Reshape to spatial tokens ---
        n_tok = h * w  # tokens_per_hand
        x = x.reshape(B, T, n_tok, C)  # (B, T, n_tok, C)

        # --- 3) Temporal 1D Conv per spatial token ---
        x = x.transpose(0, 2, 1, 3)       # (B, n_tok, T, C)
        x = x.reshape(B * n_tok, T, C)    # (B*n_tok, T, C)
        x = nn.Conv(C, kernel_size=(3,), strides=(1,), padding="SAME")(x)
        x = nn.gelu(x)
        x = nn.GroupNorm(num_groups=min(8, C))(x)
        x = nn.Conv(C, kernel_size=(3,), strides=(1,), padding="SAME")(x)
        x = nn.gelu(x)
        x = nn.GroupNorm(num_groups=min(8, C))(x)
        # (B*n_tok, T, C)

        # --- 4) Temporal aggregation ---
        x = jnp.max(x, axis=1)            # (B*n_tok, C)
        x = x.reshape(B, n_tok, C)        # (B, tokens_per_hand, C)

        # --- 5) Project to transformer width ---
        x = nn.Dense(cfg.width)(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dropout(rate=cfg.dropout)(x, deterministic=not train)

        return x  # (B, tokens_per_hand, width)


class DualFSRToTokens(nn.Module):
    """Dual FSR tactile module producing multiple tokens per hand.

    With default config: 4 tokens per hand x 2 hands = 8 tokens total.
    """

    cfg: DualFSR3DCNNConfig

    @nn.compact
    def __call__(self, fsr_left: jnp.ndarray, fsr_right: jnp.ndarray, *, train: bool):
        encoder = SharedFSR2DTemporalEncoder(self.cfg)
        tok_left = encoder(fsr_left, train=train)    # (B, tokens_per_hand, width)
        tok_right = encoder(fsr_right, train=train)  # (B, tokens_per_hand, width)
        return tok_left, tok_right
