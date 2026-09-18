#!/usr/bin/env python3
"""
Train a tactile prediction model (v2: single-frame output, conv-upsample decoder).

Input:  front image + wrist image + tactile_left(5,16,16) + tactile_right(5,16,16)
        + state(7) + action(7)
Output: predicted next tactile_left(1,16,16) + tactile_right(1,16,16)

Tactile streams use a **shared** spatial-CNN + ConvLSTM encoder (see `TactileEncoder`).
Decoder uses ConvTranspose upsampling (4×4 → 8×8 → 16×16) instead of MLP.

Reuses the LeRobot dataset infrastructure from the VLA codebase.

Usage:
    # tyro: the config name is the first positional argument; do not use --config-name.
    uv run python scripts/train_tactile_prediction.py uf850_pi05_lora_tactile_sft
    uv run python scripts/train_tactile_prediction.py uf850_pi05_lora_freeze_tactile --batch-size 32

    The script file name is train_tactile_prediction.py, not train_tactile_predictor.py.
    For overriding dataset and other fields, run:
    python scripts/train_tactile_prediction.py <config_name> --help
"""

import dataclasses
import logging
import os
import pickle
import platform
from typing import Sequence, SupportsIndex

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import tqdm_loggable.auto as tqdm
import wandb

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.training.data_loader import TransformedDataset, TorchDataLoader


# ===================================================================
# Config
# ===================================================================

@dataclasses.dataclass
class TPConfig:
    """Tactile prediction training hyperparameters."""
    # Image encoder
    image_resolution: int = 128
    image_feat_dim: int = 256
    # Tactile encoder
    tactile_T: int = 5
    tactile_H: int = 16
    tactile_W: int = 16
    tactile_feat_dim: int = 128
    # ConvLSTM tactile encoder: spatial embedding channels and LSTM hidden channels.
    tactile_embed_dim: int = 128
    tactile_lstm_hidden: int = 128
    tactile_conv_lstm_kernel: int = 3
    # State / Action
    state_dim: int = 7
    action_dim: int = 7
    state_feat_dim: int = 64
    action_feat_dim: int = 64
    # ResNet image encoder
    image_num_filters: tuple = (64, 128, 256)
    image_blocks_per_stage: int = 2
    freeze_image_backbone: bool = False
    # Fusion & decoder
    fusion_dim: int = 512
    decoder_base_ch: int = 128
    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    # W&B
    wandb_enabled: bool = True
    project_name: str = "tactile-prediction"
    exp_name: str = "tactile_pred_v1"
    # Checkpoint
    checkpoint_dir: str = "checkpoints/tactile_pred"
    save_interval: int = 5000
    log_interval: int = 100
    # Train / val / test splits are made by full episode_index to avoid neighboring-frame leakage.
    # The three fractions should sum to < 1. Remaining episodes are used for training.
    # If both validation and test fractions are 0, no split is used and all data is used for training.
    val_episode_frac: float = 0.1
    test_episode_frac: float = 0.0
    split_seed: int = 0
    eval_interval: int = 500
    eval_batches: int = 50
    # Number of initial frames to skip per episode; approach-phase frames often have no contact and all-zero tactile data.
    skip_first_n_frames: int = 70


# ===================================================================
# Model
# ===================================================================

class ResBlock(nnx.Module):
    """Basic residual block: conv3×3 → ReLU → conv3×3 + skip."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_ch, out_ch, (3, 3), strides=(stride, stride), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, (3, 3), padding="SAME", rngs=rngs)
        self.need_proj = stride != 1 or in_ch != out_ch
        if self.need_proj:
            self.skip_proj = nnx.Conv(in_ch, out_ch, (1, 1), strides=(stride, stride), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = nnx.relu(self.conv1(x))
        x = self.conv2(x)
        if self.need_proj:
            residual = self.skip_proj(residual)
        return nnx.relu(x + residual)


class ImageResNet(nnx.Module):
    """ResNet-style image encoder with residual blocks.

    Architecture (with default filters=(64, 128, 256), input 128×128):
        stem 7×7 stride-2 → 64×64
        stage 0: 2× ResBlock(64→64)          64×64
        stage 1: 2× ResBlock(64→128, s2)     32×32
        stage 2: 2× ResBlock(128→256, s2)    16×16
        → global average pool → LayerNorm → linear proj

    When freeze_backbone=True the backbone features are stop_gradient-ed
    before projection, matching ConRFT's critic encoder behaviour.
    """

    def __init__(
        self,
        out_dim: int,
        num_filters: Sequence[int] = (64, 128, 256),
        blocks_per_stage: int = 2,
        *,
        freeze_backbone: bool = False,
        rngs: nnx.Rngs,
    ):
        self.freeze_backbone = freeze_backbone
        self.stem = nnx.Conv(3, num_filters[0], (7, 7), strides=(2, 2), padding="SAME", rngs=rngs)

        self.blocks = []
        in_ch = num_filters[0]
        for stage_idx, out_ch in enumerate(num_filters):
            for block_idx in range(blocks_per_stage):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                self.blocks.append(ResBlock(in_ch, out_ch, stride=stride, rngs=rngs))
                in_ch = out_ch

        self.norm = nnx.LayerNorm(num_filters[-1], rngs=rngs)
        self.proj = nnx.Linear(num_filters[-1], out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.stem(x))
        for block in self.blocks:
            x = block(x)
        x = jnp.mean(x, axis=(1, 2))
        if self.freeze_backbone:
            x = jax.lax.stop_gradient(x)
        return self.proj(self.norm(x))


class ConvLSTMCell(nnx.Module):
    """Single-step ConvLSTM in NHWC layout; see Shi et al., Convolutional LSTM Network."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        *,
        kernel_size: int = 3,
        rngs: nnx.Rngs,
    ):
        k = kernel_size
        self.hidden_channels = hidden_channels
        self.conv_x = nnx.Conv(in_channels, 4 * hidden_channels, (k, k), padding="SAME", rngs=rngs)
        self.conv_h = nnx.Conv(hidden_channels, 4 * hidden_channels, (k, k), padding="SAME", rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        h: jnp.ndarray,
        c: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        gates = self.conv_x(x) + self.conv_h(h)
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        o = jax.nn.sigmoid(o)
        g = jnp.tanh(g)
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return h_new, c_new


class TactileEncoder(nnx.Module):
    """Per-frame 2D CNN embedding + temporal ConvLSTM + spatial pooling + linear projection.

    (B, T, H, W) -> shared per-frame CNN -> (B, T, h', w', C_emb)
    -> run ConvLSTM over T with jax.lax.scan -> take final h_T
    -> spatial mean pooling -> LayerNorm -> Linear(out_dim)
    """

    def __init__(
        self,
        out_dim: int,
        *,
        embed_dim: int = 128,
        lstm_hidden: int = 128,
        conv_lstm_kernel: int = 3,
        rngs: nnx.Rngs,
    ):
        # Same three-stage spatial downsampling as the previous version: 16x16 -> 2x2, channels -> embed_dim.
        self.c1 = nnx.Conv(1, 32, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c2 = nnx.Conv(32, 64, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c3 = nnx.Conv(64, embed_dim, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.cell = ConvLSTMCell(
            embed_dim,
            lstm_hidden,
            kernel_size=conv_lstm_kernel,
            rngs=rngs,
        )
        self.embed_dim = embed_dim
        self.lstm_hidden = lstm_hidden
        self.norm = nnx.LayerNorm(lstm_hidden, rngs=rngs)
        self.proj = nnx.Linear(lstm_hidden, out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, H, W = x.shape
        z = x.reshape(B * T, H, W, 1)
        for conv in (self.c1, self.c2, self.c3):
            z = nnx.relu(conv(z))
        _, hh, ww, _ = z.shape
        z = z.reshape(B, T, hh, ww, self.embed_dim)
        # (T, B, hh, ww, C) for scan.
        z = jnp.transpose(z, (1, 0, 2, 3, 4))

        h0 = jnp.zeros((B, hh, ww, self.lstm_hidden), dtype=z.dtype)
        c0 = jnp.zeros((B, hh, ww, self.lstm_hidden), dtype=z.dtype)

        def step(carry: tuple[jnp.ndarray, jnp.ndarray], x_t: jnp.ndarray):
            h, c = carry
            h_new, c_new = self.cell(x_t, h, c)
            return (h_new, c_new), None

        (h_final, _), _ = jax.lax.scan(step, (h0, c0), z)
        # Convert final-step spatial features to a vector.
        h_final = jnp.mean(h_final, axis=(1, 2))
        return self.proj(self.norm(h_final))


class TactileDecoder(nnx.Module):
    """Conv-upsample decoder: latent vector → single (1, 16, 16) tactile frame.

    512 → Linear(base_ch*4*4) → reshape [B,4,4,base_ch]
    → ConvTranspose ×2 (4→8→16) → Conv head → [B,1,16,16]
    """

    def __init__(self, in_dim: int, base_ch: int = 128, *, rngs: nnx.Rngs):
        self.base_ch = base_ch
        self.fc = nnx.Linear(in_dim, base_ch * 4 * 4, rngs=rngs)
        self.up1 = nnx.ConvTranspose(
            base_ch, base_ch // 2, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs,
        )
        self.up2 = nnx.ConvTranspose(
            base_ch // 2, base_ch // 4, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs,
        )
        self.head = nnx.Conv(base_ch // 4, 1, (3, 3), padding="SAME", rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B = x.shape[0]
        x = nnx.relu(self.fc(x)).reshape(B, 4, 4, self.base_ch)
        x = nnx.relu(self.up1(x))
        x = nnx.relu(self.up2(x))
        x = self.head(x)                    # [B, 16, 16, 1]
        return x.transpose(0, 3, 1, 2)      # → [B, 1, 16, 16]


class TactilePredictionNet(nnx.Module):
    """Multi-modal encoder–decoder for next-tactile prediction.

    Encodes (front_img, wrist_img, tactile_left, tactile_right, state, action)
    and decodes predicted (next_tactile_left, next_tactile_right).
    """

    def __init__(self, cfg: TPConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        rk = rngs.params()
        keys = jax.random.split(rk, 11)

        self.front_enc = ImageResNet(
            cfg.image_feat_dim, cfg.image_num_filters, cfg.image_blocks_per_stage,
            freeze_backbone=cfg.freeze_image_backbone, rngs=nnx.Rngs(params=keys[0]),
        )
        self.wrist_enc = ImageResNet(
            cfg.image_feat_dim, cfg.image_num_filters, cfg.image_blocks_per_stage,
            freeze_backbone=cfg.freeze_image_backbone, rngs=nnx.Rngs(params=keys[1]),
        )
        self.tac_enc = TactileEncoder(
            cfg.tactile_feat_dim,
            embed_dim=cfg.tactile_embed_dim,
            lstm_hidden=cfg.tactile_lstm_hidden,
            conv_lstm_kernel=cfg.tactile_conv_lstm_kernel,
            rngs=nnx.Rngs(params=keys[2]),
        )
        self.state_fc1 = nnx.Linear(cfg.state_dim, cfg.state_feat_dim, rngs=nnx.Rngs(params=keys[3]))
        self.state_fc2 = nnx.Linear(cfg.state_feat_dim, cfg.state_feat_dim, rngs=nnx.Rngs(params=keys[4]))
        self.action_fc1 = nnx.Linear(cfg.action_dim, cfg.action_feat_dim, rngs=nnx.Rngs(params=keys[5]))
        self.action_fc2 = nnx.Linear(cfg.action_feat_dim, cfg.action_feat_dim, rngs=nnx.Rngs(params=keys[6]))

        total = 2 * cfg.image_feat_dim + 2 * cfg.tactile_feat_dim + cfg.state_feat_dim + cfg.action_feat_dim
        self.fuse = nnx.Linear(total, cfg.fusion_dim, rngs=nnx.Rngs(params=keys[7]))
        self.fuse_norm = nnx.LayerNorm(cfg.fusion_dim, rngs=nnx.Rngs(params=keys[8]))

        self.dec_l = TactileDecoder(cfg.fusion_dim, cfg.decoder_base_ch, rngs=nnx.Rngs(params=keys[9]))
        self.dec_r = TactileDecoder(cfg.fusion_dim, cfg.decoder_base_ch, rngs=nnx.Rngs(params=keys[10]))

    def __call__(
        self,
        front_img: jnp.ndarray,
        wrist_img: jnp.ndarray,
        tac_left: jnp.ndarray,
        tac_right: jnp.ndarray,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            front_img:  [B, H, W, 3] float32 in [-1, 1]
            wrist_img:  [B, H, W, 3] float32 in [-1, 1]
            tac_left:   [B, 5, 16, 16] float32 (preprocessed)
            tac_right:  [B, 5, 16, 16] float32 (preprocessed)
            state:      [B, 7] float32
            action:     [B, 7] float32
        Returns:
            (pred_left, pred_right): each [B, 1, 16, 16]
        """
        feat = jnp.concatenate([
            self.front_enc(front_img),
            self.wrist_enc(wrist_img),
            self.tac_enc(tac_left),
            self.tac_enc(tac_right),
            nnx.relu(self.state_fc2(nnx.relu(self.state_fc1(state)))),
            nnx.relu(self.action_fc2(nnx.relu(self.action_fc1(action)))),
        ], axis=-1)
        feat = self.fuse_norm(nnx.relu(self.fuse(feat)))
        return self.dec_l(feat), self.dec_r(feat)


# ===================================================================
# Dataset & Transforms
# ===================================================================

def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _parse_image(img, resolution: int) -> np.ndarray:
    """Convert dataset image to float32 HWC in [-1, 1], resized to `resolution`."""
    from PIL import Image as PILImage

    img = _to_np(img)
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = einops.rearrange(img, "c h w -> h w c")
    pil = PILImage.fromarray(img.astype(np.uint8))
    pil = pil.resize((resolution, resolution), PILImage.BILINEAR)
    return np.array(pil, dtype=np.float32) / 127.5 - 1.0


class TactileTransitionDataset:
    """Wraps a per-frame dataset and produces (current_obs, next_tactile) pairs.

    At episode boundaries the next-tactile target is a copy of the current tactile
    and `valid_mask` is set to 0 so the loss can ignore it.
    """

    def __init__(self, base, *, episode_key: str = "episode_index"):
        self._base = base
        self._ek = episode_key

    def __len__(self) -> int:
        return max(0, len(self._base) - 1)

    def __getitem__(self, idx: SupportsIndex) -> dict:
        i = idx.__index__()
        s = self._base[i]
        s_next = self._base[i + 1]

        ep_cur = s.get(self._ek, 0)
        ep_nxt = s_next.get(self._ek, 0)
        if isinstance(ep_cur, torch.Tensor):
            ep_cur = ep_cur.item()
        if isinstance(ep_nxt, torch.Tensor):
            ep_nxt = ep_nxt.item()
        valid = int(ep_cur) == int(ep_nxt)

        out = dict(s)
        if valid:
            out["next_tactile_left"] = s_next.get(
                "observation/tactile_left", s.get("observation/tactile_left")
            )
            out["next_tactile_right"] = s_next.get(
                "observation/tactile_right", s.get("observation/tactile_right")
            )
        else:
            out["next_tactile_left"] = s.get("observation/tactile_left")
            out["next_tactile_right"] = s.get("observation/tactile_right")
        out["valid_mask"] = np.array(float(valid), dtype=np.float32)
        return out


class TactilePredFinalTransform:
    """Extracts model-ready arrays from a transition sample dict.

    Resizes images, slices state/action to the expected dimensions,
    and applies TactilePreprocess to next-frame tactile targets.
    """

    def __init__(
        self,
        image_resolution: int = 128,
        state_dim: int = 7,
        action_dim: int = 7,
    ):
        self.res = image_resolution
        self.sd = state_dim
        self.ad = action_dim
        self._tactile_tf = _transforms.TactilePreprocess()

    def __call__(self, sample: dict) -> dict:
        out = {}
        out["front_image"] = _parse_image(sample["observation/image"], self.res)
        out["wrist_image"] = _parse_image(sample["observation/wrist_image"], self.res)

        state = _to_np(sample["observation/state"]).astype(np.float32).flatten()
        out["state"] = state[: self.sd]

        action = _to_np(sample.get("actions", sample.get("action"))).astype(np.float32)
        if action.ndim == 2:
            action = action[0]
        out["action"] = action[: self.ad]

        for key in ("observation/tactile_left", "observation/tactile_right"):
            short = key.split("/", 1)[1]
            v = sample.get(key)
            out[short] = _to_np(v).astype(np.float32) if v is not None else np.zeros(
                (5, 16, 16), dtype=np.float32
            )

        for src_key, out_key in [
            ("next_tactile_left", "target_tactile_left"),
            ("next_tactile_right", "target_tactile_right"),
        ]:
            v = sample.get(src_key)
            if v is not None:
                arr = _to_np(v).astype(np.float32)
                arr = arr[-1:] if arr.ndim >= 3 else arr.reshape(1, 16, 16)
                out[out_key] = arr
            else:
                out[out_key] = np.zeros((1, 16, 16), dtype=np.float32)

        out["valid_mask"] = _to_np(sample.get("valid_mask", 1.0)).astype(np.float32)
        return out


class IndexSubsetDataset:
    """Wrap a dataset with a subset of transition indices from `TactileTransitionDataset`."""

    def __init__(self, base, indices: np.ndarray):
        self._base = base
        self._indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: SupportsIndex) -> dict:
        return self._base[int(self._indices[idx.__index__()])]


def _get_frame_episode_indices(raw_dataset) -> np.ndarray:
    """Return episode_index values aligned with raw_dataset frames; length equals len(raw_dataset)."""
    hf = getattr(raw_dataset, "hf_dataset", None)
    if hf is not None:
        cols = getattr(hf, "column_names", []) or []
        if "episode_index" in cols:
            col = hf["episode_index"]
            if hasattr(col, "numpy"):
                return np.asarray(col.numpy(), dtype=np.int64)
            return np.asarray(col, dtype=np.int64)
    n = len(raw_dataset)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        row = raw_dataset[i]
        ei = row.get("episode_index", 0)
        if isinstance(ei, torch.Tensor):
            ei = ei.item()
        out[i] = int(ei)
    return out


def _compute_within_episode_index(episode_per_frame: np.ndarray) -> np.ndarray:
    """Compute the 0-based within-episode frame index for each frame."""
    within = np.zeros_like(episode_per_frame)
    prev_ep = -1
    counter = 0
    for i in range(len(episode_per_frame)):
        ep = int(episode_per_frame[i])
        if ep != prev_ep:
            counter = 0
            prev_ep = ep
        within[i] = counter
        counter += 1
    return within


def _split_transition_indices_by_episode(
    episode_per_frame: np.ndarray,
    val_frac: float,
    test_frac: float,
    seed: int,
    skip_first_n: int = 0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
    """Split transition indices into train/val/test by the episode of the start frame.

    Transition ``i`` uses the episode_id of frame ``i``, aligned with LeRobot frames.
    When ``skip_first_n > 0``, transitions from the first N frames of each episode are dropped.
    """
    n_frames = int(episode_per_frame.shape[0])
    if n_frames < 2:
        return np.array([], dtype=np.int64), None, None, {}

    within_ep = _compute_within_episode_index(episode_per_frame)
    keep_mask = within_ep[:-1] >= skip_first_n
    if skip_first_n > 0:
        n_dropped = int((~keep_mask).sum())
        logging.info("skip_first_n_frames=%d -> dropped %d transitions (%.1f%%)",
                     skip_first_n, n_dropped, 100.0 * n_dropped / max(len(keep_mask), 1))

    unique_eps = np.unique(episode_per_frame)
    if val_frac <= 0.0 and test_frac <= 0.0:
        train_idx = np.where(keep_mask)[0].astype(np.int64)
        return train_idx, None, None, {int(e): "train" for e in unique_eps}

    assert val_frac >= 0.0 and test_frac >= 0.0
    assert val_frac + test_frac < 1.0 - 1e-9, "val_episode_frac + test_episode_frac must be less than 1"

    rng = np.random.default_rng(seed)
    eps_shuffled = unique_eps.copy()
    rng.shuffle(eps_shuffled)
    n_ep = len(eps_shuffled)
    n_val = int(round(n_ep * val_frac))
    n_test = int(round(n_ep * test_frac))
    n_train = n_ep - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"No training episodes remain after splitting ({n_ep} total episodes). Reduce the val/test fractions."
        )

    train_eps = set(eps_shuffled[:n_train].tolist())
    val_eps = set(eps_shuffled[n_train : n_train + n_val].tolist()) if n_val > 0 else set()
    test_eps = set(eps_shuffled[n_train + n_val :].tolist()) if n_test > 0 else set()

    ep_split = {}
    for e in train_eps:
        ep_split[int(e)] = "train"
    for e in val_eps:
        ep_split[int(e)] = "val"
    for e in test_eps:
        ep_split[int(e)] = "test"

    train_idx, val_idx, test_idx = [], [], []
    for i in range(n_frames - 1):
        if not keep_mask[i]:
            continue
        e = int(episode_per_frame[i])
        if e in train_eps:
            train_idx.append(i)
        elif e in val_eps:
            val_idx.append(i)
        elif e in test_eps:
            test_idx.append(i)

    train_arr = np.asarray(train_idx, dtype=np.int64)
    val_arr = np.asarray(val_idx, dtype=np.int64) if val_idx else None
    test_arr = np.asarray(test_idx, dtype=np.int64) if test_idx else None

    logging.info(
        "Episode split: total_episodes=%d  train=%d  val=%d  test=%d | transitions: train=%d  val=%s  test=%s",
        n_ep, len(train_eps), len(val_eps), len(test_eps),
        len(train_arr), len(val_arr) if val_arr is not None else 0,
        len(test_arr) if test_arr is not None else 0,
    )
    return train_arr, val_arr, test_arr, ep_split


# ===================================================================
# Training
# ===================================================================

@nnx.jit
def train_step(
    model: TactilePredictionNet,
    optimizer: nnx.Optimizer,
    batch: dict,
) -> tuple[jnp.ndarray, dict]:
    def loss_fn(model):
        pred_l, pred_r = model(
            batch["front_image"],
            batch["wrist_image"],
            batch["tactile_left"],
            batch["tactile_right"],
            batch["state"],
            batch["action"],
        )
        tgt_l = batch["target_tactile_left"]
        tgt_r = batch["target_tactile_right"]
        mask = batch["valid_mask"]

        per_sample_l = jnp.mean((pred_l - tgt_l) ** 2, axis=(1, 2, 3))
        per_sample_r = jnp.mean((pred_r - tgt_r) ** 2, axis=(1, 2, 3))
        n_valid = jnp.maximum(mask.sum(), 1.0)
        mse_l = jnp.sum(mask * per_sample_l) / n_valid
        mse_r = jnp.sum(mask * per_sample_r) / n_valid
        loss = mse_l + mse_r
        return loss, {
            "loss": loss,
            "mse_left": mse_l,
            "mse_right": mse_r,
            "valid_frac": mask.mean(),
        }

    (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)
    return loss, info


@nnx.jit
def eval_step(model: TactilePredictionNet, batch: dict) -> tuple[jnp.ndarray, dict]:
    """Forward pass and loss only for validation/testing, without gradients."""

    pred_l, pred_r = model(
        batch["front_image"],
        batch["wrist_image"],
        batch["tactile_left"],
        batch["tactile_right"],
        batch["state"],
        batch["action"],
    )
    tgt_l = batch["target_tactile_left"]
    tgt_r = batch["target_tactile_right"]
    mask = batch["valid_mask"]

    per_sample_l = jnp.mean((pred_l - tgt_l) ** 2, axis=(1, 2, 3))
    per_sample_r = jnp.mean((pred_r - tgt_r) ** 2, axis=(1, 2, 3))
    n_valid = jnp.maximum(mask.sum(), 1.0)
    mse_l = jnp.sum(mask * per_sample_l) / n_valid
    mse_r = jnp.sum(mask * per_sample_r) / n_valid
    loss = mse_l + mse_r
    return loss, {
        "loss": loss,
        "mse_left": mse_l,
        "mse_right": mse_r,
        "valid_frac": mask.mean(),
    }


def _average_eval_metrics(model: TactilePredictionNet, loader: TorchDataLoader, num_batches: int) -> dict:
    """Take ``num_batches`` batches from loader and average eval_step metrics."""
    it = iter(loader)
    acc = {}
    for _ in range(num_batches):
        batch = next(it)
        _, info = eval_step(model, batch)
        info_cpu = jax.device_get(info)
        for k, v in info_cpu.items():
            acc[k] = acc.get(k, 0.0) + float(v)
    return {k: acc[k] / max(num_batches, 1) for k in acc}


# ===================================================================
# Helpers
# ===================================================================

def init_logging():
    level_map = {
        "DEBUG": "D", "INFO": "I", "WARNING": "W",
        "ERROR": "E", "CRITICAL": "C",
    }

    class Fmt(logging.Formatter):
        def format(self, record):
            record.levelname = level_map.get(record.levelname, record.levelname)
            return super().format(record)

    fmt = Fmt(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s "
        "(%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(fmt)
    else:
        logger.addHandler(logging.StreamHandler())
        logger.handlers[0].setFormatter(fmt)


def save_checkpoint(model, step: int, ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:08d}.pkl")
    state = {
        "step": step,
        "params": jax.device_get(nnx.state(model).to_pure_dict()),
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)
    logging.info("Saved checkpoint → %s", path)


def load_latest_checkpoint(model, ckpt_dir: str) -> int:
    if not os.path.isdir(ckpt_dir):
        return 0
    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pkl"))
    if not files:
        return 0
    path = os.path.join(ckpt_dir, files[-1])
    logging.info("Loading checkpoint ← %s", path)
    with open(path, "rb") as f:
        state = pickle.load(f)
    ms = nnx.state(model)
    ms.replace_by_pure_dict(state["params"])
    nnx.update(model, ms)
    logging.info("Restored step %d", state["step"])
    return state["step"]


# ===================================================================
# Main
# ===================================================================

def main(train_config: _config.TrainConfig):
    tp = TPConfig()
    init_logging()
    logging.info("Platform: %s", platform.node())
    logging.info("TPConfig: %s", dataclasses.asdict(tp))
    logging.info("batch_size=%d  num_train_steps=%d  seed=%d",
                 train_config.batch_size, train_config.num_train_steps, train_config.seed)

    rng = jax.random.key(train_config.seed)

    # ---- Data ----
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    repo_id = data_config.repo_id
    logging.info("Loading dataset: %s", repo_id)

    _dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    raw_dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={"action": [0.0]},
    )
    # Do not use hf_dataset.with_format("torch").
    # LeRobot calls `torch.stack` on the result of HF `select(q_idx)` in __getitem__;
    # for single-frame queries, some versions return a single Tensor instead of a tensor list, causing:
    #   TypeError: stack(): argument 'tensors' must be tuple of Tensors, not Tensor
    # This reliably appears in DataLoader workers when num_workers > 0.
    # Keep the default format, usually NumPy, and let collate stack the batch.
    _hf = getattr(raw_dataset, "hf_dataset", None)
    if _hf is not None:
        logging.info("LeRobot hf_dataset: %s (not forcing torch format for worker safety)", type(_hf).__name__)

    repack = _transforms.RepackTransform({
        "observation/image": "observation.images.front",
        "observation/wrist_image": "observation.images.wrist",
        "observation/state": "observation.state",
        "observation/tactile_left": "observation.tactile_left",
        "observation/tactile_right": "observation.tactile_right",
        "actions": "action",
        "episode_index": "episode_index",
    })
    tactile_tf = _transforms.TactilePreprocess()
    transformed = TransformedDataset(raw_dataset, [repack, tactile_tf])

    transition_ds = TactileTransitionDataset(transformed, episode_key="episode_index")
    logging.info("Dataset frames: %d  transitions: %d", len(raw_dataset), len(transition_ds))

    final_ds_full = TransformedDataset(
        transition_ds,
        [TactilePredFinalTransform(tp.image_resolution, tp.state_dim, tp.action_dim)],
    )

    local_batch = train_config.batch_size // jax.process_count()

    episode_per_frame = _get_frame_episode_indices(raw_dataset)
    train_trans_idx, val_trans_idx, test_trans_idx, _ = _split_transition_indices_by_episode(
        episode_per_frame,
        tp.val_episode_frac,
        tp.test_episode_frac,
        tp.split_seed,
        skip_first_n=tp.skip_first_n_frames,
    )

    if tp.val_episode_frac <= 0.0 and tp.test_episode_frac <= 0.0:
        train_ds = final_ds_full
        val_loader = None
        test_loader = None
    else:
        train_ds = IndexSubsetDataset(final_ds_full, train_trans_idx)
        val_loader = None
        test_loader = None
        if val_trans_idx is not None and len(val_trans_idx) > 0:
            val_ds = IndexSubsetDataset(final_ds_full, val_trans_idx)
            if len(val_ds) >= train_config.batch_size:
                val_loader = TorchDataLoader(
                    val_ds,
                    local_batch_size=local_batch,
                    shuffle=False,
                    num_workers=train_config.num_workers,
                    seed=train_config.seed + 1,
                )
            else:
                logging.warning(
                    "Validation transition count (%d) < batch_size (%d); skipping validation DataLoader.",
                    len(val_trans_idx),
                    train_config.batch_size,
                )
        if test_trans_idx is not None and len(test_trans_idx) > 0:
            test_ds = IndexSubsetDataset(final_ds_full, test_trans_idx)
            if len(test_ds) >= train_config.batch_size:
                test_loader = TorchDataLoader(
                    test_ds,
                    local_batch_size=local_batch,
                    shuffle=False,
                    num_workers=train_config.num_workers,
                    seed=train_config.seed + 2,
                )
            else:
                logging.warning(
                    "Test transition count (%d) < batch_size (%d); skipping test DataLoader.",
                    len(test_trans_idx),
                    train_config.batch_size,
                )

    if len(train_ds) < local_batch:
        raise ValueError(
            f"Training transition count ({len(train_ds)}) is smaller than batch_size ({local_batch})."
            "Reduce batch_size or disable/reduce validation and test splits."
        )

    loader = TorchDataLoader(
        train_ds,
        local_batch_size=local_batch,
        shuffle=True,
        num_workers=train_config.num_workers,
        seed=train_config.seed,
    )
    data_iter = iter(loader)
    batch0 = next(data_iter)
    logging.info("Batch shapes: %s", {k: np.shape(v) for k, v in batch0.items()})

    # ---- Model ----
    model = TactilePredictionNet(tp, rngs=nnx.Rngs(params=rng))
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model)))
    logging.info("Model parameters: %d (%.2f M)", n_params, n_params / 1e6)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=tp.lr,
        warmup_steps=tp.warmup_steps,
        decay_steps=train_config.num_train_steps,
        end_value=tp.lr * 0.01,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(tp.grad_clip),
        optax.adamw(schedule, weight_decay=tp.weight_decay),
    )
    optimizer = nnx.Optimizer(model, tx)

    start_step = load_latest_checkpoint(model, tp.checkpoint_dir)

    # ---- W&B ----
    if tp.wandb_enabled:
        wandb.init(
            name=tp.exp_name,
            project=tp.project_name,
            config={**dataclasses.asdict(tp), "repo_id": repo_id,
                    "batch_size": train_config.batch_size},
        )
    else:
        wandb.init(mode="disabled")

    # Log a few sample images
    front_imgs = batch0["front_image"]
    vis = [wandb.Image(np.clip((front_imgs[i] + 1) * 127.5, 0, 255).astype(np.uint8))
           for i in range(min(4, front_imgs.shape[0]))]
    wandb.log({"train/sample_images": vis}, step=0)

    # ---- Training loop ----
    pbar = tqdm.tqdm(
        range(start_step, train_config.num_train_steps),
        initial=start_step,
        total=train_config.num_train_steps,
        dynamic_ncols=True,
    )
    logging.info("Training from step %d to %d", start_step, train_config.num_train_steps - 1)

    for step in pbar:
        batch = batch0 if step == start_step else next(data_iter)
        loss, info = train_step(model, optimizer, batch)

        if step % tp.log_interval == 0:
            info_cpu = jax.device_get(info)
            pbar.set_postfix({k: f"{float(v):.4f}" for k, v in info_cpu.items()})
            wandb.log({f"train/{k}": float(v) for k, v in info_cpu.items()}, step=step)

        if val_loader is not None and step % tp.eval_interval == 0:
            val_metrics = _average_eval_metrics(model, val_loader, tp.eval_batches)
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
            logging.info("val metrics @ %d: %s", step, val_metrics)

        if (step > start_step and step % tp.save_interval == 0) or step == train_config.num_train_steps - 1:
            save_checkpoint(model, step, tp.checkpoint_dir)

    if test_loader is not None:
        test_metrics = _average_eval_metrics(model, test_loader, tp.eval_batches)
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()}, step=train_config.num_train_steps)
        logging.info("Test metrics (hold-out episodes): %s", test_metrics)

    logging.info("Training complete.")


if __name__ == "__main__":
    main(_config.cli())
