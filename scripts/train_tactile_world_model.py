#!/usr/bin/env python3
"""Tactile World Model — two-stage training.

Stage 1  VAE pretraining
   Input : single tactile block  X_k ∈ R^{5×16×16}
   Output: reconstruction         X̂_k ∈ R^{5×16×16}
   Loss  : λ_l1·L1 + λ_l2·L2 + β·KL
   若 W&B 里 KL 很快贴 0、rec 仍很低，多为后验坍塌；可调大 ``vae_kl_beta``、设
   ``vae_kl_anneal_steps``（KL 退火）或 ``vae_kl_free_bits``（见 ``WMConfig``）。

Stage 2  Forward dynamics
   Input : H history latents  z_{k−H+1:k}  +  H actions  a_{k−H+1:k}
   Output: predicted next block  X̂_{k+1} ∈ R^{5×16×16}
   Loss  : pixel (L1+L2) + latent alignment

Data assumption:
   action at 30 Hz, each action step corresponds to 5 tactile sub-frames.
   Each dataset frame already stores tactile_left / tactile_right as (5,16,16).

Usage:
   脚本文件名是 ``train_tactile_world_model.py``（world 与 model 之间有下划线）。

   openpi 的 ``TrainConfig`` 要求 ``--exp-name``；若命令行未提供，本脚本会自动补上
   ``--exp-name tactile_world_model``（也可用环境变量 ``EXP_NAME`` 覆盖默认值）。

   # Stage 1 — VAE pretraining
   STAGE=1 uv run python scripts/train_tactile_world_model.py uf850_pi05_lora_tactile_sft

   # 显式指定实验名（与 assets/checkpoint 子目录一致）
   STAGE=1 uv run python scripts/train_tactile_world_model.py uf850_pi05_lora_tactile_sft \\
       --exp-name my_twm_run

   # Stage 2 — dynamics (loads best VAE from Stage 1)
   STAGE=2 VAE_CKPT=checkpoints/tactile_wm/vae/step_00010000.pkl \\
       uv run python scripts/train_tactile_world_model.py uf850_pi05_lora_tactile_sft

   # Stage 2 — dynamics with visual (wrist image + state) conditioning
   STAGE=2 USE_VISUAL=1 VAE_CKPT=checkpoints/tactile_wm/vae/step_00010000.pkl \\
       uv run python scripts/train_tactile_world_model.py uf850_pi05_lora_tactile_sft
   # Optional（与 DataConfig repack 一致）: WRIST_IMAGE_KEY=... STATE_KEY=observation.state
   # state 维度默认 7，且会在 USE_VISUAL=1 时从首帧自动推断；也可 STATE_DIM=14 强制指定

   # Stage 3 — 仅可视化：加载 Stage2 dynamics + 冻结 VAE，在验证集选一个 episode，
   # 对「每个合法中心 k」画下一 block 的真值 vs 预测（需与训练时相同的 split_seed / H）。
   STAGE=3 DYN_CKPT=checkpoints/tactile_wm/dynamics/step_00010000.pkl \\
       uv run python scripts/train_tactile_world_model.py uf850_pi05_lora_tactile_sft
   # 可选：VAE_CKPT=path/to/vae.pkl 或目录；EPISODE_ID=12 强制指定验证集里的 episode_index
"""

import dataclasses
import logging
import os
import sys
import pickle
import platform
from typing import SupportsIndex

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
import tqdm_loggable.auto as tqdm
import wandb

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.training.data_loader import TransformedDataset, TorchDataLoader

# Visual encoder — reuse pretrained ResNet-10 architecture from ConRFT RL script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_tacrl_offline import (
    PreTrainedResNetImageEncoder,
    _download_resnet10_params,
    _map_linen_to_nnx_backbone,
)


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class WMConfig:
    """Tactile world-model hyper-parameters (both stages)."""

    # --- stage selection (override via env STAGE=1|2) ---
    stage: int = 1

    # --- tactile block shape ---
    tactile_T: int = 5
    tactile_H: int = 16
    tactile_W: int = 16

    # --- VAE architecture ---
    latent_dim: int = 64
    enc_channels: tuple = (32, 64, 128)   # per-frame CNN channel progression
    enc_gru_hidden: int = 128             # temporal GRU hidden size
    dec_init_spatial: int = 4             # spatial size before ConvTranspose
    dec_init_ch: int = 64                 # channel width entering decoder

    # --- VAE loss (Stage 1) ---
    vae_l1_w: float = 0.7
    vae_l2_w: float = 0.3
    # KL 权重；过小易出现「KL→0、潜变量不用」的后验坍塌。
    vae_kl_beta: float = 0.02
    # KL 线性退火：前 N step 将 β 从 0 升到 vae_kl_beta；0 表示关闭（恒定 β）。
    vae_kl_anneal_steps: int = 8000
    # Free-bits（每维最小 KL，单位 nat）；0 表示关闭。例如 0.02～0.05 可缓解 KL 被压死。
    vae_kl_free_bits: float = 0.02

    # --- history predictor (Stage 2) ---
    history_len: int = 3
    action_dim: int = 7
    action_embed_dim: int = 32
    hist_gru_hidden: int = 128
    future_head_dim: int = 256

    # --- Stage 2 loss ---
    future_l1_w: float = 1.0
    future_l2_w: float = 0.3
    latent_loss_w: float = 1.0

    # --- VAE checkpoint path for Stage 2 (override via env VAE_CKPT) ---
    vae_ckpt: str = ""

    # --- optimizer ---
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # --- logging ---
    wandb_enabled: bool = True
    project_name: str = "tactile-world-model"
    exp_name: str = "twm_v1"

    # --- checkpoint ---
    ckpt_dir: str = "checkpoints/tactile_wm"
    save_interval: int = 5000
    log_interval: int = 100

    # --- data split ---
    val_frac: float = 0.1
    test_frac: float = 0.0
    split_seed: int = 0
    eval_interval: int = 500
    eval_batches: int = 50
    skip_first_n: int = 70

    # --- visual + state conditioning (Stage 2) ---
    use_visual: bool = False
    # 与 ``DataConfig`` 里 ``observation/wrist_image`` → ``observation.images.wrist`` 一致
    wrist_image_key: str = "observation.images.wrist"
    state_key: str = "observation.state"
    # 须与 ``observation.state`` 展平后长度一致（常见 7）；也可用环境变量 STATE_DIM 或 main 里自动推断
    state_dim: int = 7
    image_out_dim: int = 256
    proprio_latent_dim: int = 64
    num_spatial_blocks: int = 8


# ═══════════════════════════════════════════════════════════════════
# Model — building blocks
# ═══════════════════════════════════════════════════════════════════

class GRUCell(nnx.Module):
    """Standard GRU cell for use inside jax.lax.scan."""

    def __init__(self, in_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim
        self.W_xz = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.W_hz = nnx.Linear(hidden_dim, hidden_dim, use_bias=False, rngs=rngs)
        self.W_xr = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.W_hr = nnx.Linear(hidden_dim, hidden_dim, use_bias=False, rngs=rngs)
        self.W_xn = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.W_hn = nnx.Linear(hidden_dim, hidden_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        z = jax.nn.sigmoid(self.W_xz(x) + self.W_hz(h))
        r = jax.nn.sigmoid(self.W_xr(x) + self.W_hr(h))
        n = jnp.tanh(self.W_xn(x) + r * self.W_hn(h))
        return (1 - z) * n + z * h


# ═══════════════════════════════════════════════════════════════════
# Model — Block VAE  (Stage 1)
# ═══════════════════════════════════════════════════════════════════

class BlockVAEEncoder(nnx.Module):
    """(B,5,16,16) → per-frame shared CNN → temporal GRU → (μ, logσ²).

    Architecture:
        per-frame CNN (16→8→4→2, channels 32→64→128) → flatten 512
        → [B,5,512] → GRU(hidden=128) → h_final [B,128]
        → Linear → μ [B,Z]
        → Linear → logvar [B,Z]
    """

    def __init__(self, cfg: WMConfig, *, rngs: nnx.Rngs):
        chs = cfg.enc_channels
        self.c1 = nnx.Conv(1, chs[0], (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c2 = nnx.Conv(chs[0], chs[1], (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c3 = nnx.Conv(chs[1], chs[2], (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        # 16/(2^3) = 2 → flat = 128*2*2 = 512
        cnn_flat = chs[2] * (cfg.tactile_H // 8) * (cfg.tactile_W // 8)
        self.gru_cell = GRUCell(cnn_flat, cfg.enc_gru_hidden, rngs=rngs)
        self.gru_hidden = cfg.enc_gru_hidden
        self.fc_mu = nnx.Linear(cfg.enc_gru_hidden, cfg.latent_dim, rngs=rngs)
        self.fc_logvar = nnx.Linear(cfg.enc_gru_hidden, cfg.latent_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """x: [B, 5, 16, 16] → (mu, logvar) each [B, Z]."""
        B, T, H, W = x.shape
        z = x.reshape(B * T, H, W, 1)
        z = nnx.relu(self.c1(z))
        z = nnx.relu(self.c2(z))
        z = nnx.relu(self.c3(z))
        z = z.reshape(B, T, -1)                     # [B, T, 512]
        z_seq = jnp.transpose(z, (1, 0, 2))         # [T, B, 512]
        h0 = jnp.zeros((B, self.gru_hidden))

        def step(h, x_t):
            return self.gru_cell(x_t, h), None

        h_final, _ = jax.lax.scan(step, h0, z_seq)  # [B, 128]
        return self.fc_mu(h_final), self.fc_logvar(h_final)


class BlockVAEDecoder(nnx.Module):
    """z [B,Z] → Linear → [B*5,4,4,ch] → per-frame ConvTranspose → [B,5,16,16].

    Architecture:
        z → Linear(Z, 5*4*4*64=5120) → reshape [B*5, 4, 4, 64]
        → ConvT 64→32 4×4 s2 → [B*5, 8, 8, 32]
        → ConvT 32→16 4×4 s2 → [B*5, 16, 16, 16]
        → Conv  16→1  3×3    → [B*5, 16, 16, 1]
        → reshape [B, 5, 16, 16]
    """

    def __init__(self, cfg: WMConfig, *, rngs: nnx.Rngs):
        sp, ch, T = cfg.dec_init_spatial, cfg.dec_init_ch, cfg.tactile_T
        self.T, self.sp, self.ch = T, sp, ch
        self.fc = nnx.Linear(cfg.latent_dim, T * sp * sp * ch, rngs=rngs)
        self.up1 = nnx.ConvTranspose(ch, ch // 2, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.up2 = nnx.ConvTranspose(ch // 2, ch // 4, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.head = nnx.Conv(ch // 4, 1, (3, 3), padding="SAME", rngs=rngs)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """z: [B, Z] → [B, 5, 16, 16]."""
        B = z.shape[0]
        x = nnx.relu(self.fc(z))
        x = x.reshape(B * self.T, self.sp, self.sp, self.ch)
        x = nnx.relu(self.up1(x))
        x = nnx.relu(self.up2(x))
        x = self.head(x)                            # [B*T, 16, 16, 1]
        return x.reshape(B, self.T, 16, 16)


class BlockVAE(nnx.Module):
    """Tactile block VAE = encoder + reparameterize + decoder."""

    def __init__(self, cfg: WMConfig, *, rngs: nnx.Rngs):
        keys = jax.random.split(rngs.params(), 2)
        self.encoder = BlockVAEEncoder(cfg, rngs=nnx.Rngs(params=keys[0]))
        self.decoder = BlockVAEDecoder(cfg, rngs=nnx.Rngs(params=keys[1]))

    def encode(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.encoder(x)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)

    def __call__(
        self, x: jnp.ndarray, *, rng_key: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass with reparameterization.

        Returns (x_recon, mu, logvar, z).
        """
        mu, logvar = self.encode(x)
        logvar = jnp.clip(logvar, -10.0, 10.0)
        std = jnp.exp(0.5 * logvar)
        z = mu + std * jax.random.normal(rng_key, mu.shape)
        return self.decode(z), mu, logvar, z


# ═══════════════════════════════════════════════════════════════════
# Model — History Predictor  (Stage 2)
# ═══════════════════════════════════════════════════════════════════

class ActionEncoder(nnx.Module):
    """Two-layer MLP: in_dim → out_dim with ReLU (used for action and state in the visual predictor)."""

    def __init__(self, in_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, out_dim, rngs=rngs)
        self.fc2 = nnx.Linear(out_dim, out_dim, rngs=rngs)

    def __call__(self, a: jnp.ndarray) -> jnp.ndarray:
        return nnx.relu(self.fc2(nnx.relu(self.fc1(a))))


class HistoryPredictor(nnx.Module):
    """(z_hist [B,H,Z], a_hist [B,H,7]) → z_future [B,Z].

    1. action encoder  7→32
    2. concat [z_i, e_i] → [B, H, Z+A]
    3. GRU(hidden=128) over H → h_k [B,128]
    4. future head 128→256→Z
    """

    def __init__(self, cfg: WMConfig, *, rngs: nnx.Rngs):
        keys = jax.random.split(rngs.params(), 4)
        self.action_enc = ActionEncoder(
            cfg.action_dim, cfg.action_embed_dim, rngs=nnx.Rngs(params=keys[0]),
        )
        in_dim = cfg.latent_dim + cfg.action_embed_dim
        self.gru_cell = GRUCell(in_dim, cfg.hist_gru_hidden, rngs=nnx.Rngs(params=keys[1]))
        self.gru_hidden = cfg.hist_gru_hidden
        self.head1 = nnx.Linear(cfg.hist_gru_hidden, cfg.future_head_dim, rngs=nnx.Rngs(params=keys[2]))
        self.head2 = nnx.Linear(cfg.future_head_dim, cfg.latent_dim, rngs=nnx.Rngs(params=keys[3]))

    def __call__(self, z_hist: jnp.ndarray, a_hist: jnp.ndarray) -> jnp.ndarray:
        B, H, Z = z_hist.shape
        e = self.action_enc(a_hist.reshape(B * H, -1)).reshape(B, H, -1)
        u = jnp.concatenate([z_hist, e], axis=-1)     # [B, H, Z+A]
        u_seq = jnp.transpose(u, (1, 0, 2))           # [H, B, Z+A]
        h0 = jnp.zeros((B, self.gru_hidden))

        def step(h, u_t):
            return self.gru_cell(u_t, h), None

        h_final, _ = jax.lax.scan(step, h0, u_seq)    # [B, 128]
        return self.head2(nnx.relu(self.head1(h_final)))


class HistoryPredictorWithVision(nnx.Module):
    """History predictor conditioned on wrist image + proprio state.

    Reuses the same pretrained ResNet-10 image encoder architecture as the
    critic encoder in ``train_tacrl_offline.py``.  State uses the same
    MLP style as action: ``ActionEncoder`` (Linear→ReLU→Linear→ReLU).

    Per-step feature = [z_tactile, vis_feat, state_feat, action_embed]
    → GRU over H history steps → future head → z_future.
    """

    def __init__(self, cfg: WMConfig, *, rngs: nnx.Rngs):
        keys = jax.random.split(rngs.params(), 6)

        self.action_enc = ActionEncoder(
            cfg.action_dim, cfg.action_embed_dim, rngs=nnx.Rngs(params=keys[0]),
        )

        self.vis_encoder = PreTrainedResNetImageEncoder(
            out_dim=cfg.image_out_dim,
            num_spatial_blocks=cfg.num_spatial_blocks,
            rngs=nnx.Rngs(params=keys[1]),
        )

        self.state_enc = ActionEncoder(
            cfg.state_dim, cfg.proprio_latent_dim, rngs=nnx.Rngs(params=keys[2]),
        )

        in_dim = cfg.latent_dim + cfg.image_out_dim + cfg.proprio_latent_dim + cfg.action_embed_dim
        self.gru_cell = GRUCell(in_dim, cfg.hist_gru_hidden, rngs=nnx.Rngs(params=keys[3]))
        self.gru_hidden = cfg.hist_gru_hidden

        self.head1 = nnx.Linear(cfg.hist_gru_hidden, cfg.future_head_dim, rngs=nnx.Rngs(params=keys[4]))
        self.head2 = nnx.Linear(cfg.future_head_dim, cfg.latent_dim, rngs=nnx.Rngs(params=keys[5]))

    def _encode_context(
        self,
        wrist_hist: jnp.ndarray,
        state_hist: jnp.ndarray,
        a_hist: jnp.ndarray,
    ) -> jnp.ndarray:
        """Encode visual + state + action → [B, H, context_dim]."""
        B, H = a_hist.shape[:2]

        vis_flat = wrist_hist.reshape(B * H, *wrist_hist.shape[2:])
        vis_feat = self.vis_encoder(vis_flat, train=False).reshape(B, H, -1)

        s_flat = state_hist.reshape(B * H, -1)
        s_feat = self.state_enc(s_flat).reshape(B, H, -1)

        a_flat = a_hist.reshape(B * H, -1)
        a_feat = self.action_enc(a_flat).reshape(B, H, -1)

        return jnp.concatenate([vis_feat, s_feat, a_feat], axis=-1)

    def _predict(self, z_hist: jnp.ndarray, context: jnp.ndarray) -> jnp.ndarray:
        """z_hist [B,H,Z] + context [B,H,C] → z_future [B,Z]."""
        u = jnp.concatenate([z_hist, context], axis=-1)
        u_seq = jnp.transpose(u, (1, 0, 2))
        B = z_hist.shape[0]
        h0 = jnp.zeros((B, self.gru_hidden))

        def step(h, u_t):
            return self.gru_cell(u_t, h), None

        h_final, _ = jax.lax.scan(step, h0, u_seq)
        return self.head2(nnx.relu(self.head1(h_final)))

    def __call__(
        self,
        zl_hist: jnp.ndarray,
        zr_hist: jnp.ndarray,
        a_hist: jnp.ndarray,
        wrist_hist: jnp.ndarray,
        state_hist: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Predict next left and right tactile latents.

        Returns ``(zl_future, zr_future)`` each ``[B, Z]``.
        Context (visual + state + action) is encoded once and shared.
        """
        ctx = self._encode_context(wrist_hist, state_hist, a_hist)
        return self._predict(zl_hist, ctx), self._predict(zr_hist, ctx)


# ═══════════════════════════════════════════════════════════════════
# Data pipeline
# ═══════════════════════════════════════════════════════════════════

def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class BlockTransform:
    """Extracts model-ready tactile blocks + action (+ optional wrist image / state)."""

    def __init__(self, action_dim: int = 7, *, extract_visual: bool = False):
        self.ad = action_dim
        self.extract_visual = extract_visual

    def __call__(self, sample: dict) -> dict:
        out: dict = {}
        for key in ("observation/tactile_left", "observation/tactile_right"):
            short = key.split("/", 1)[1]
            v = sample.get(key)
            out[short] = (
                _to_np(v).astype(np.float32) if v is not None
                else np.zeros((5, 16, 16), dtype=np.float32)
            )
        action = _to_np(sample.get("actions", sample.get("action"))).astype(np.float32)
        if action.ndim == 2:
            action = action[0]
        out["action"] = action[: self.ad]

        if self.extract_visual:
            img = sample.get("wrist_image")
            if img is not None:
                img = _to_np(img).astype(np.float32)
                # LeRobot: [C, H, W] in [0,1] → [H, W, C] in [-1, 1]
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))
                if img.max() <= 1.0 + 1e-6:
                    img = img * 2.0 - 1.0
                out["wrist_image"] = img

            state = sample.get("state")
            if state is not None:
                out["state"] = _to_np(state).astype(np.float32).flatten()

        return out


class IndexSubset:
    """Random-access subset of a dataset, selected by index array."""

    def __init__(self, base, indices: np.ndarray):
        self._base = base
        self._idx = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, idx: SupportsIndex) -> dict:
        return self._base[int(self._idx[idx.__index__()])]


class SequenceDataset:
    """Returns (H history blocks + actions, target next block) for Stage 2.

    ``center_indices[i]`` is the frame index *k* such that:
        history  = frames k−H+1 … k
        target   = frame  k+1
    """

    def __init__(self, base, center_indices: np.ndarray, history_len: int):
        self._base = base
        self._idx = np.asarray(center_indices, dtype=np.int64)
        self._H = history_len

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, idx: SupportsIndex) -> dict:
        k = int(self._idx[idx.__index__()])
        H = self._H
        frames = [self._base[k - H + 1 + i] for i in range(H + 1)]
        out = {
            "tactile_left_hist": np.stack([f["tactile_left"] for f in frames[:H]]),
            "tactile_right_hist": np.stack([f["tactile_right"] for f in frames[:H]]),
            "action_hist": np.stack([f["action"] for f in frames[:H]]),
            "target_tactile_left": frames[H]["tactile_left"],
            "target_tactile_right": frames[H]["tactile_right"],
        }
        if "wrist_image" in frames[0]:
            out["wrist_image_hist"] = np.stack([f["wrist_image"] for f in frames[:H]])
        if "state" in frames[0]:
            out["state_hist"] = np.stack([f["state"] for f in frames[:H]])
        return out


# ═══════════════════════════════════════════════════════════════════
# Episode / index helpers
# ═══════════════════════════════════════════════════════════════════

def _get_frame_episodes(raw_dataset) -> np.ndarray:
    """Per-frame episode_index aligned with raw_dataset, shape (N,)."""
    hf = getattr(raw_dataset, "hf_dataset", None)
    if hf is not None:
        cols = getattr(hf, "column_names", []) or []
        if "episode_index" in cols:
            col = hf["episode_index"]
            return np.asarray(col.numpy() if hasattr(col, "numpy") else col, dtype=np.int64)
    n = len(raw_dataset)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        ei = raw_dataset[i].get("episode_index", 0)
        out[i] = int(ei.item() if isinstance(ei, torch.Tensor) else ei)
    return out


def _within_episode_idx(ep: np.ndarray) -> np.ndarray:
    """0-based position of each frame inside its episode."""
    w = np.zeros_like(ep)
    prev, cnt = -1, 0
    for i in range(len(ep)):
        e = int(ep[i])
        if e != prev:
            cnt, prev = 0, e
        w[i] = cnt
        cnt += 1
    return w


def _valid_frame_indices(ep: np.ndarray, skip_first_n: int = 0) -> np.ndarray:
    """Indices of frames whose within-episode position ≥ skip_first_n."""
    return np.where(_within_episode_idx(ep) >= skip_first_n)[0].astype(np.int64)


def _valid_sequence_indices(ep: np.ndarray, H: int, skip_first_n: int = 0) -> np.ndarray:
    """Valid centre-k values for (k−H+1 … k+1) windows that stay within one episode."""
    within = _within_episode_idx(ep)
    n = len(ep)
    if n <= H:
        return np.array([], dtype=np.int64)

    # run[i] = length of consecutive same-episode run ending at i
    same = np.concatenate([[False], ep[1:] == ep[:-1]])
    run = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        run[i] = (run[i - 1] + 1) if same[i] else 0

    candidates = np.arange(H - 1, n - 1)
    mask = (run[candidates + 1] >= H) & (within[candidates - H + 1] >= skip_first_n)
    result = candidates[mask].astype(np.int64)
    logging.info("Valid sequences (H=%d, skip=%d): %d / %d candidates", H, skip_first_n, len(result), len(candidates))
    return result


def _split_by_episode(
    indices: np.ndarray,
    ep: np.ndarray,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Split indices into train / val / test by episode."""
    if val_frac <= 0 and test_frac <= 0:
        return indices, None, None

    eps = ep[indices]
    unique = np.unique(eps)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    n = len(unique)
    nv = int(round(n * val_frac))
    nt = int(round(n * test_frac))
    train_set = set(unique[: n - nv - nt].tolist())
    val_set = set(unique[n - nv - nt : n - nt].tolist())
    test_set = set(unique[n - nt :].tolist())

    train_mask = np.isin(eps, list(train_set))
    val_mask = np.isin(eps, list(val_set)) if val_set else np.zeros(len(eps), dtype=bool)
    test_mask = np.isin(eps, list(test_set)) if test_set else np.zeros(len(eps), dtype=bool)

    train_idx = indices[train_mask]
    val_idx = indices[val_mask] if val_mask.any() else None
    test_idx = indices[test_mask] if test_mask.any() else None

    logging.info(
        "Episode split → train=%d  val=%s  test=%s  (eps: %d / %d / %d)",
        len(train_idx),
        len(val_idx) if val_idx is not None else 0,
        len(test_idx) if test_idx is not None else 0,
        len(train_set), len(val_set), len(test_set),
    )
    return train_idx, val_idx, test_idx


# ═══════════════════════════════════════════════════════════════════
# Checkpoint / logging helpers
# ═══════════════════════════════════════════════════════════════════

def _init_logging():
    tags = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class _Fmt(logging.Formatter):
        def format(self, record):
            record.levelname = tags.get(record.levelname, record.levelname)
            return super().format(record)

    fmt = _Fmt(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s "
        "(%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(fmt)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger.addHandler(handler)


def _save_ckpt(model: nnx.Module, step: int, ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:08d}.pkl")
    state = {"step": step, "params": jax.device_get(nnx.state(model).to_pure_dict())}
    with open(path, "wb") as f:
        pickle.dump(state, f)
    logging.info("Saved → %s", path)


def _load_ckpt(model: nnx.Module, path: str) -> int:
    """Load checkpoint from exact path. Returns restored step, or 0 on failure."""
    if not os.path.isfile(path):
        logging.warning("Checkpoint not found: %s", path)
        return 0
    logging.info("Loading ← %s", path)
    with open(path, "rb") as f:
        state = pickle.load(f)
    ms = nnx.state(model)
    ms.replace_by_pure_dict(state["params"])
    nnx.update(model, ms)
    logging.info("Restored step %d", state["step"])
    return state["step"]


def _load_latest_ckpt(model: nnx.Module, ckpt_dir: str) -> int:
    """Scan ckpt_dir for the latest .pkl and load it."""
    if not os.path.isdir(ckpt_dir):
        return 0
    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pkl"))
    if not files:
        return 0
    return _load_ckpt(model, os.path.join(ckpt_dir, files[-1]))


def _vae_kl_per_dim(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """Diagonal Gaussian KL to N(0,I), shape (*, Z)."""
    return -0.5 * (1.0 + logvar - mu**2 - jnp.exp(logvar))


def _vae_kl_scalar(mu: jnp.ndarray, logvar: jnp.ndarray, free_bits_per_dim: float) -> jnp.ndarray:
    per = _vae_kl_per_dim(mu, logvar)
    if free_bits_per_dim > 0:
        per = jnp.maximum(per, free_bits_per_dim)
    return jnp.mean(per)


# ═══════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════

def _visualize_vae_reconstruction(
    vae: "BlockVAE",
    batch: dict,
    step: int,
    rng_key: jnp.ndarray,
    *,
    save_dir: str = "",
    n_samples: int = 4,
    hand: str = "left",
) -> list:
    """Generate reconstruction comparison figures for W&B + local save.

    For each sample produces a figure with 3 rows × 5 cols:
        row 0 — original  (5 sub-frames)
        row 1 — reconstructed
        row 2 — |error|
    Returns a list of wandb.Image objects.
    """
    key = f"tactile_{hand}"
    x = batch[key]                                       # [B, 5, 16, 16]
    n = min(n_samples, x.shape[0])
    x_sub = x[:n]
    recon, _, _, _ = vae(x_sub, rng_key=rng_key)
    x_np = np.asarray(jax.device_get(x_sub))             # [n, 5, 16, 16]
    r_np = np.asarray(jax.device_get(recon))

    vmin = float(min(x_np.min(), r_np.min()))
    vmax = float(max(x_np.max(), r_np.max()))

    images = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(n):
        fig, axes = plt.subplots(3, 5, figsize=(15, 8))
        err = np.abs(x_np[i] - r_np[i])
        emax = float(max(err.max(), 1e-8))

        for t in range(5):
            axes[0, t].imshow(x_np[i, t], cmap="hot", vmin=vmin, vmax=vmax, aspect="equal")
            axes[0, t].set_title(f"orig t={t}", fontsize=9)
            axes[0, t].axis("off")

            axes[1, t].imshow(r_np[i, t], cmap="hot", vmin=vmin, vmax=vmax, aspect="equal")
            axes[1, t].set_title(f"recon t={t}", fontsize=9)
            axes[1, t].axis("off")

            im_err = axes[2, t].imshow(err[t], cmap="hot", vmin=0, vmax=emax, aspect="equal")
            axes[2, t].set_title(f"|err| t={t}", fontsize=9)
            axes[2, t].axis("off")

        fig.colorbar(im_err, ax=axes[2, :].tolist(), fraction=0.02, pad=0.02, label="|error|")

        mse_val = float(np.mean(err ** 2))
        mae_val = float(np.mean(err))
        fig.suptitle(
            f"step={step} | {hand} sample {i} | MSE={mse_val:.5f}  MAE={mae_val:.5f}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save_dir:
            path = os.path.join(save_dir, f"step{step:08d}_{hand}_s{i}.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")

        images.append(wandb.Image(fig, caption=f"{hand} sample {i}"))
        plt.close(fig)

    return images


def _save_episode_reconstruction(
    vae: "BlockVAE",
    base_ds,
    val_idx: np.ndarray,
    ep: np.ndarray,
    step: int,
    rng_key: jnp.ndarray,
    *,
    save_dir: str,
    hand: str = "left",
) -> None:
    """Save orig / recon / error for every frame of one validation episode.

    Picks a validation episode whose frames are closest to the middle of
    the dataset (most likely to contain contact), then saves one PNG per
    frame **for the entire episode** (ignoring skip_first_n so that the
    full trajectory is visible).
    Also saves a timeline summary PNG showing MAE per frame.
    """
    val_eps = ep[val_idx]
    unique_eps = np.unique(val_eps)
    if len(unique_eps) == 0:
        return

    # pick episode whose median frame index is closest to the dataset midpoint
    dataset_mid = len(ep) // 2
    best_ep, best_dist = int(unique_eps[0]), float("inf")
    for ue in unique_eps:
        frames_of_ep = val_idx[val_eps == ue]
        median_frame = int(np.median(frames_of_ep))
        d = abs(median_frame - dataset_mid)
        if d < best_dist:
            best_dist, best_ep = d, int(ue)

    # use ALL frames of the chosen episode (not just val_idx subset)
    ep_frames = np.where(ep == best_ep)[0].astype(np.int64)
    ep_frames = np.sort(ep_frames)
    n_frames = len(ep_frames)
    if n_frames == 0:
        return

    out_dir = os.path.join(save_dir, f"ep{best_ep}_{hand}")
    os.makedirs(out_dir, exist_ok=True)
    logging.info("Saving full episode reconstruction: ep=%d  frames=%d  hand=%s → %s",
                 best_ep, n_frames, hand, out_dir)

    key = f"tactile_{hand}"
    maes = []
    mses = []

    for fi, frame_idx in enumerate(ep_frames):
        sample = base_ds[int(frame_idx)]
        x = np.asarray(sample[key], dtype=np.float32)       # [5, 16, 16]
        x_jax = jnp.expand_dims(jnp.asarray(x), 0)         # [1, 5, 16, 16]
        rng_key, sk = jax.random.split(rng_key)
        recon, _, _, _ = vae(x_jax, rng_key=sk)
        r_np = np.asarray(jax.device_get(recon[0]))          # [5, 16, 16]

        err = np.abs(x - r_np)
        mae_val = float(np.mean(err))
        mse_val = float(np.mean(err ** 2))
        maes.append(mae_val)
        mses.append(mse_val)

        vmin = float(min(x.min(), r_np.min()))
        vmax = float(max(x.max(), r_np.max()))
        emax = float(max(err.max(), 1e-8))

        fig, axes = plt.subplots(3, 5, figsize=(15, 8))
        for t in range(5):
            axes[0, t].imshow(x[t], cmap="hot", vmin=vmin, vmax=vmax, aspect="equal")
            axes[0, t].set_title(f"orig t={t}", fontsize=9)
            axes[0, t].axis("off")

            axes[1, t].imshow(r_np[t], cmap="hot", vmin=vmin, vmax=vmax, aspect="equal")
            axes[1, t].set_title(f"recon t={t}", fontsize=9)
            axes[1, t].axis("off")

            im_e = axes[2, t].imshow(err[t], cmap="hot", vmin=0, vmax=emax, aspect="equal")
            axes[2, t].set_title(f"|err| t={t}", fontsize=9)
            axes[2, t].axis("off")

        fig.colorbar(im_e, ax=axes[2, :].tolist(), fraction=0.02, pad=0.02, label="|error|")
        fig.suptitle(
            f"ep={best_ep} frame={fi}/{n_frames} (idx={frame_idx}) | "
            f"{hand} | MAE={mae_val:.5f}  MSE={mse_val:.5f}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(out_dir, f"frame_{fi:04d}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # timeline summary: MAE per frame
    fig_sum, ax_sum = plt.subplots(figsize=(12, 4))
    ax_sum.plot(maes, label="MAE", color="tab:red")
    ax_sum.plot(mses, label="MSE", color="tab:blue", alpha=0.7)
    ax_sum.set_xlabel("Frame in episode")
    ax_sum.set_ylabel("Error")
    ax_sum.set_title(f"ep={best_ep} | {hand} | step={step} | {n_frames} frames")
    ax_sum.legend()
    ax_sum.grid(True, alpha=0.3)
    fig_sum.tight_layout()
    fig_sum.savefig(os.path.join(out_dir, "timeline_error.png"), dpi=120)
    plt.close(fig_sum)

    logging.info("Episode %d %s: mean MAE=%.5f  mean MSE=%.5f",
                 best_ep, hand, float(np.mean(maes)), float(np.mean(mses)))


# ═══════════════════════════════════════════════════════════════════
# Stage 1 — VAE pretraining
# ═══════════════════════════════════════════════════════════════════

def _run_stage1(cfg: WMConfig, tc: _config.TrainConfig, base_ds, ep: np.ndarray, local_batch: int, rng):
    logging.info("═══ Stage 1: Block-VAE pretraining ═══")

    # --- data ---
    valid = _valid_frame_indices(ep, cfg.skip_first_n)
    train_idx, val_idx, _ = _split_by_episode(valid, ep, cfg.val_frac, cfg.test_frac, cfg.split_seed)

    train_ds = IndexSubset(base_ds, train_idx)
    logging.info("Train samples: %d", len(train_ds))

    loader = TorchDataLoader(
        train_ds, local_batch_size=local_batch, shuffle=True,
        num_workers=tc.num_workers, seed=tc.seed,
    )
    val_loader = None
    if val_idx is not None and len(val_idx) >= tc.batch_size:
        val_loader = TorchDataLoader(
            IndexSubset(base_ds, val_idx), local_batch_size=local_batch,
            shuffle=False, num_workers=tc.num_workers, seed=tc.seed + 1,
        )

    # 预取一个"中间帧"验证 batch，用于固定可视化（避免 episode 头尾全零帧）
    fixed_vis_batch = None
    if val_idx is not None and len(val_idx) >= local_batch:
        mid_start = len(val_idx) // 3
        mid_indices = val_idx[mid_start : mid_start + local_batch]
        if len(mid_indices) == local_batch:
            mid_ds = IndexSubset(base_ds, mid_indices)
            from openpi.training.data_loader import _collate_fn
            fixed_vis_batch = _collate_fn([mid_ds[i] for i in range(local_batch)])
            fixed_vis_batch = jax.tree.map(np.asarray, fixed_vis_batch)
            logging.info("Fixed vis batch: val indices [%d:%d] (mid-episode region)",
                         mid_start, mid_start + local_batch)

    # --- model ---
    vae = BlockVAE(cfg, rngs=nnx.Rngs(params=rng))
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(vae)))
    logging.info("BlockVAE parameters: %d (%.2f M)", n_params, n_params / 1e6)

    schedule = optax.warmup_cosine_decay_schedule(
        0.0, cfg.lr, cfg.warmup_steps, tc.num_train_steps, cfg.lr * 0.01,
    )
    tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adamw(schedule, weight_decay=cfg.weight_decay))
    optimizer = nnx.Optimizer(vae, tx)

    vae_dir = os.path.join(cfg.ckpt_dir, "vae")
    start_step = _load_latest_ckpt(vae, vae_dir)

    # --- wandb ---
    if cfg.wandb_enabled:
        wandb.init(name=f"{cfg.exp_name}_vae", project=cfg.project_name,
                   config={**dataclasses.asdict(cfg), "batch_size": tc.batch_size})
    else:
        wandb.init(mode="disabled")

    # --- jit-compiled steps (capture loss weights via closure) ---
    l1w, l2w = cfg.vae_l1_w, cfg.vae_l2_w
    beta_max = cfg.vae_kl_beta
    anneal_steps = cfg.vae_kl_anneal_steps
    fb = cfg.vae_kl_free_bits

    @nnx.jit
    def train_step(vae_m, opt, batch, step_key, step_i):
        step_f = jnp.asarray(step_i, dtype=jnp.float32)
        denom = jnp.maximum(jnp.asarray(float(anneal_steps), dtype=jnp.float32), 1.0)
        beta_eff = jnp.where(
            anneal_steps > 0,
            jnp.minimum(1.0, (step_f + 1.0) / denom) * beta_max,
            beta_max,
        )

        def loss_fn(vae_m):
            tl, tr = batch["tactile_left"], batch["tactile_right"]
            k1, k2 = jax.random.split(step_key)
            rl, mul, lvl, _ = vae_m(tl, rng_key=k1)
            rr, mur, lvr, _ = vae_m(tr, rng_key=k2)

            rec = (l1w * (jnp.mean(jnp.abs(rl - tl)) + jnp.mean(jnp.abs(rr - tr)))
                   + l2w * (jnp.mean((rl - tl) ** 2) + jnp.mean((rr - tr) ** 2)))
            kl = _vae_kl_scalar(mul, lvl, fb) + _vae_kl_scalar(mur, lvr, fb)
            loss = rec + beta_eff * kl
            return loss, {
                "loss": loss,
                "rec": rec,
                "kl": kl,
                "beta_eff": beta_eff,
                "kl_weighted": beta_eff * kl,
            }

        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(vae_m)
        opt.update(grads)
        return loss, info

    @nnx.jit
    def eval_step(vae_m, batch, step_key):
        """验证用完整 β（不参与退火），与最终目标一致。"""
        tl, tr = batch["tactile_left"], batch["tactile_right"]
        k1, k2 = jax.random.split(step_key)
        rl, mul, lvl, _ = vae_m(tl, rng_key=k1)
        rr, mur, lvr, _ = vae_m(tr, rng_key=k2)
        rec = (l1w * (jnp.mean(jnp.abs(rl - tl)) + jnp.mean(jnp.abs(rr - tr)))
               + l2w * (jnp.mean((rl - tl) ** 2) + jnp.mean((rr - tr) ** 2)))
        kl = _vae_kl_scalar(mul, lvl, fb) + _vae_kl_scalar(mur, lvr, fb)
        return rec + beta_max * kl, {"loss": rec + beta_max * kl, "rec": rec, "kl": kl}

    # --- training loop ---
    data_iter = iter(loader)
    rng_loop = rng
    pbar = tqdm.tqdm(range(start_step, tc.num_train_steps), initial=start_step, total=tc.num_train_steps, dynamic_ncols=True)

    for step in pbar:
        batch = next(data_iter)
        rng_loop, sk = jax.random.split(rng_loop)
        _, info = train_step(vae, optimizer, batch, sk, step)

        if step % cfg.log_interval == 0:
            ic = jax.device_get(info)
            pbar.set_postfix({k: f"{float(v):.4f}" for k, v in ic.items()})
            wandb.log({f"train/{k}": float(v) for k, v in ic.items()}, step=step)

        if val_loader is not None and step > 0 and step % cfg.eval_interval == 0:
            val_it = iter(val_loader)
            acc: dict[str, float] = {}
            vis_batch = None
            for bi in range(cfg.eval_batches):
                vb = next(val_it)
                if bi == 0:
                    vis_batch = vb
                rng_loop, vk = jax.random.split(rng_loop)
                _, vi = eval_step(vae, vb, vk)
                for k, v in jax.device_get(vi).items():
                    acc[k] = acc.get(k, 0.0) + float(v)
            val_m = {k: v / cfg.eval_batches for k, v in acc.items()}
            wandb.log({f"val/{k}": v for k, v in val_m.items()}, step=step)
            logging.info("val @ %d: %s", step, val_m)

            log_dict: dict = {}
            vis_dir = os.path.join(vae_dir, "vis")

            # fixed mid-episode samples — 跨 step 对比重建质量变化
            if fixed_vis_batch is not None:
                rng_loop, vk_l, vk_r = jax.random.split(rng_loop, 3)
                log_dict["val/recon_left"] = _visualize_vae_reconstruction(
                    vae, fixed_vis_batch, step, vk_l, save_dir=vis_dir, n_samples=4, hand="left",
                )
                log_dict["val/recon_right"] = _visualize_vae_reconstruction(
                    vae, fixed_vis_batch, step, vk_r, save_dir=vis_dir, n_samples=4, hand="right",
                )

            # random samples — 覆盖更多样本，抽一个随机 batch
            if vis_batch is not None:
                try:
                    rng_loop, rk = jax.random.split(rng_loop)
                    rand_skip = int(jax.random.randint(rk, (), 1, max(cfg.eval_batches, 2)))
                    rand_it = iter(val_loader)
                    for _ in range(rand_skip):
                        rand_batch = next(rand_it)
                    rng_loop, rk_l, rk_r = jax.random.split(rng_loop, 3)
                    log_dict["val/recon_left_rand"] = _visualize_vae_reconstruction(
                        vae, rand_batch, step, rk_l, n_samples=2, hand="left",
                    )
                    log_dict["val/recon_right_rand"] = _visualize_vae_reconstruction(
                        vae, rand_batch, step, rk_r, n_samples=2, hand="right",
                    )
                except StopIteration:
                    pass

            if log_dict:
                wandb.log(log_dict, step=step)

        if (step > start_step and step % cfg.save_interval == 0) or step == tc.num_train_steps - 1:
            _save_ckpt(vae, step, vae_dir)

    # 训练结束后保存一次完整 episode 的重建结果（左右手）
    if val_idx is not None and len(val_idx) > 0:
        ep_vis_dir = os.path.join(vae_dir, f"episode_vis_step{tc.num_train_steps}")
        rng_loop, ek_l, ek_r = jax.random.split(rng_loop, 3)
        _save_episode_reconstruction(
            vae, base_ds, val_idx, ep, tc.num_train_steps, ek_l,
            save_dir=ep_vis_dir, hand="left",
        )
        _save_episode_reconstruction(
            vae, base_ds, val_idx, ep, tc.num_train_steps, ek_r,
            save_dir=ep_vis_dir, hand="right",
        )

    logging.info("Stage 1 complete.")


# ═══════════════════════════════════════════════════════════════════
# Stage 2 — Forward dynamics
# ═══════════════════════════════════════════════════════════════════

def _run_stage2(cfg: WMConfig, tc: _config.TrainConfig, base_ds, ep: np.ndarray, local_batch: int, rng):
    logging.info("═══ Stage 2: Forward dynamics ═══")
    H = cfg.history_len

    # --- data ---
    valid = _valid_sequence_indices(ep, H, cfg.skip_first_n)
    if len(valid) == 0:
        raise RuntimeError(f"No valid sequences found (H={H}, skip={cfg.skip_first_n}). "
                           "Reduce history_len or skip_first_n.")
    train_idx, val_idx, _ = _split_by_episode(valid, ep, cfg.val_frac, cfg.test_frac, cfg.split_seed)

    train_ds = SequenceDataset(base_ds, train_idx, H)
    logging.info("Train sequences: %d", len(train_ds))

    loader = TorchDataLoader(
        train_ds, local_batch_size=local_batch, shuffle=True,
        num_workers=tc.num_workers, seed=tc.seed,
    )
    val_loader = None
    if val_idx is not None and len(val_idx) >= tc.batch_size:
        val_loader = TorchDataLoader(
            SequenceDataset(base_ds, val_idx, H), local_batch_size=local_batch,
            shuffle=False, num_workers=tc.num_workers, seed=tc.seed + 1,
        )

    # --- load frozen VAE ---
    vae = BlockVAE(cfg, rngs=nnx.Rngs(params=rng))
    vae_path = cfg.vae_ckpt or os.path.join(cfg.ckpt_dir, "vae")
    if os.path.isdir(vae_path):
        _load_latest_ckpt(vae, vae_path)
    else:
        _load_ckpt(vae, vae_path)
    logging.info("VAE loaded and frozen for Stage 2.")

    # --- predictor ---
    use_visual = cfg.use_visual
    rng, pk = jax.random.split(rng)
    if use_visual:
        predictor = HistoryPredictorWithVision(cfg, rngs=nnx.Rngs(params=pk))
        linen_params = _download_resnet10_params()
        _map_linen_to_nnx_backbone(linen_params, predictor.vis_encoder.backbone)
        logging.info("ResNet-10 pretrained weights injected into visual encoder backbone.")
    else:
        predictor = HistoryPredictor(cfg, rngs=nnx.Rngs(params=pk))
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(predictor)))
    logging.info("Predictor (%s) parameters: %d (%.2f M)",
                 type(predictor).__name__, n_params, n_params / 1e6)

    schedule = optax.warmup_cosine_decay_schedule(
        0.0, cfg.lr, cfg.warmup_steps, tc.num_train_steps, cfg.lr * 0.01,
    )
    tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adamw(schedule, weight_decay=cfg.weight_decay))
    optimizer = nnx.Optimizer(predictor, tx)

    dyn_dir = os.path.join(cfg.ckpt_dir, "dynamics_vis" if use_visual else "dynamics")
    start_step = _load_latest_ckpt(predictor, dyn_dir)

    # --- wandb ---
    suffix = "dyn_vis" if use_visual else "dyn"
    if cfg.wandb_enabled:
        wandb.init(name=f"{cfg.exp_name}_{suffix}", project=cfg.project_name,
                   config={**dataclasses.asdict(cfg), "batch_size": tc.batch_size})
    else:
        wandb.init(mode="disabled")

    fl1, fl2, lat_w = cfg.future_l1_w, cfg.future_l2_w, cfg.latent_loss_w

    # --- jit-compiled steps (use_visual is a Python bool → resolved at trace time) ---
    @nnx.jit
    def train_step(vae_m, pred, opt, batch):
        B, _H = batch["tactile_left_hist"].shape[:2]

        tl_flat = batch["tactile_left_hist"].reshape(B * _H, 5, 16, 16)
        tr_flat = batch["tactile_right_hist"].reshape(B * _H, 5, 16, 16)
        zl_hist = jax.lax.stop_gradient(vae_m.encode(tl_flat)[0].reshape(B, _H, -1))
        zr_hist = jax.lax.stop_gradient(vae_m.encode(tr_flat)[0].reshape(B, _H, -1))

        zl_tgt = jax.lax.stop_gradient(vae_m.encode(batch["target_tactile_left"])[0])
        zr_tgt = jax.lax.stop_gradient(vae_m.encode(batch["target_tactile_right"])[0])

        def loss_fn(pred):
            if use_visual:
                zl_f, zr_f = pred(zl_hist, zr_hist, batch["action_hist"],
                                   batch["wrist_image_hist"], batch["state_hist"])
            else:
                zl_f = pred(zl_hist, batch["action_hist"])
                zr_f = pred(zr_hist, batch["action_hist"])

            pl = vae_m.decode(zl_f)
            pr = vae_m.decode(zr_f)
            tgt_l, tgt_r = batch["target_tactile_left"], batch["target_tactile_right"]
            l1 = jnp.mean(jnp.abs(pl - tgt_l)) + jnp.mean(jnp.abs(pr - tgt_r))
            l2 = jnp.mean((pl - tgt_l) ** 2) + jnp.mean((pr - tgt_r) ** 2)
            future_loss = fl1 * l1 + fl2 * l2

            latent_loss = jnp.mean((zl_f - zl_tgt) ** 2) + jnp.mean((zr_f - zr_tgt) ** 2)

            loss = future_loss + lat_w * latent_loss
            return loss, {"loss": loss, "fut_l1": l1, "fut_l2": l2, "latent": latent_loss}

        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(pred)
        opt.update(grads)
        return loss, info

    @nnx.jit
    def eval_step(vae_m, pred, batch):
        B, _H = batch["tactile_left_hist"].shape[:2]
        tl_flat = batch["tactile_left_hist"].reshape(B * _H, 5, 16, 16)
        tr_flat = batch["tactile_right_hist"].reshape(B * _H, 5, 16, 16)
        zl_hist = jax.lax.stop_gradient(vae_m.encode(tl_flat)[0].reshape(B, _H, -1))
        zr_hist = jax.lax.stop_gradient(vae_m.encode(tr_flat)[0].reshape(B, _H, -1))
        zl_tgt = jax.lax.stop_gradient(vae_m.encode(batch["target_tactile_left"])[0])
        zr_tgt = jax.lax.stop_gradient(vae_m.encode(batch["target_tactile_right"])[0])

        if use_visual:
            zl_f, zr_f = pred(zl_hist, zr_hist, batch["action_hist"],
                               batch["wrist_image_hist"], batch["state_hist"])
        else:
            zl_f = pred(zl_hist, batch["action_hist"])
            zr_f = pred(zr_hist, batch["action_hist"])

        pl, pr = vae_m.decode(zl_f), vae_m.decode(zr_f)
        tgt_l, tgt_r = batch["target_tactile_left"], batch["target_tactile_right"]

        l1 = jnp.mean(jnp.abs(pl - tgt_l)) + jnp.mean(jnp.abs(pr - tgt_r))
        l2 = jnp.mean((pl - tgt_l) ** 2) + jnp.mean((pr - tgt_r) ** 2)
        latent = jnp.mean((zl_f - zl_tgt) ** 2) + jnp.mean((zr_f - zr_tgt) ** 2)
        loss = fl1 * l1 + fl2 * l2 + lat_w * latent
        return loss, {"loss": loss, "fut_l1": l1, "fut_l2": l2, "latent": latent}

    # --- training loop ---
    data_iter = iter(loader)
    pbar = tqdm.tqdm(range(start_step, tc.num_train_steps), initial=start_step, total=tc.num_train_steps, dynamic_ncols=True)

    for step in pbar:
        batch = next(data_iter)
        _, info = train_step(vae, predictor, optimizer, batch)

        if step % cfg.log_interval == 0:
            ic = jax.device_get(info)
            pbar.set_postfix({k: f"{float(v):.4f}" for k, v in ic.items()})
            wandb.log({f"train/{k}": float(v) for k, v in ic.items()}, step=step)

        if val_loader is not None and step > 0 and step % cfg.eval_interval == 0:
            val_it = iter(val_loader)
            acc: dict[str, float] = {}
            for _ in range(cfg.eval_batches):
                _, vi = eval_step(vae, predictor, next(val_it))
                for k, v in jax.device_get(vi).items():
                    acc[k] = acc.get(k, 0.0) + float(v)
            val_m = {k: v / cfg.eval_batches for k, v in acc.items()}
            wandb.log({f"val/{k}": v for k, v in val_m.items()}, step=step)
            logging.info("val @ %d: %s", step, val_m)

        if (step > start_step and step % cfg.save_interval == 0) or step == tc.num_train_steps - 1:
            _save_ckpt(predictor, step, dyn_dir)

    logging.info("Stage 2 complete.")


# ═══════════════════════════════════════════════════════════════════
# Stage 3 — Load checkpoints, visualize full val-episode predictions
# ═══════════════════════════════════════════════════════════════════

@nnx.jit
def _jit_dynamics_predict_pixels(
    vae_m: BlockVAE,
    pred_m: HistoryPredictor,
    tl_hist: jnp.ndarray,
    tr_hist: jnp.ndarray,
    a_hist: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """tl_hist, tr_hist: [1, H, 5, 16, 16]; a_hist: [1, H, ad] → pred pixels [1, 5, 16, 16] each."""
    B, _H = tl_hist.shape[:2]
    tl_flat = tl_hist.reshape(B * _H, 5, 16, 16)
    tr_flat = tr_hist.reshape(B * _H, 5, 16, 16)
    zl_hist = vae_m.encode(tl_flat)[0].reshape(B, _H, -1)
    zr_hist = vae_m.encode(tr_flat)[0].reshape(B, _H, -1)
    zl_f = pred_m(zl_hist, a_hist)
    zr_f = pred_m(zr_hist, a_hist)
    return vae_m.decode(zl_f), vae_m.decode(zr_f)


@nnx.jit
def _jit_dynamics_predict_pixels_vis(
    vae_m: BlockVAE,
    pred_m: HistoryPredictorWithVision,
    tl_hist: jnp.ndarray,
    tr_hist: jnp.ndarray,
    a_hist: jnp.ndarray,
    wrist_hist: jnp.ndarray,
    state_hist: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Visual variant: also takes wrist_hist [1,H,Himg,Wimg,3] and state_hist [1,H,sd]."""
    B, _H = tl_hist.shape[:2]
    tl_flat = tl_hist.reshape(B * _H, 5, 16, 16)
    tr_flat = tr_hist.reshape(B * _H, 5, 16, 16)
    zl_hist = vae_m.encode(tl_flat)[0].reshape(B, _H, -1)
    zr_hist = vae_m.encode(tr_flat)[0].reshape(B, _H, -1)
    zl_f, zr_f = pred_m(zl_hist, zr_hist, a_hist, wrist_hist, state_hist)
    return vae_m.decode(zl_f), vae_m.decode(zr_f)


def _load_dynamics_ckpt(predictor, dyn_ckpt: str) -> int:
    """Load predictor from a .pkl file or the latest .pkl in a directory."""
    if os.path.isfile(dyn_ckpt):
        return _load_ckpt(predictor, dyn_ckpt)
    if os.path.isdir(dyn_ckpt):
        return _load_latest_ckpt(predictor, dyn_ckpt)
    raise FileNotFoundError(f"DYN_CKPT not found (file or dir): {dyn_ckpt}")


def _run_stage3_episode_pred_vis(
    cfg: WMConfig,
    base_ds,
    ep: np.ndarray,
    rng,
    *,
    dyn_ckpt: str,
    episode_id: int | None = None,
):
    """Load VAE + trained HistoryPredictor; save per-step prediction vs GT for one val episode."""
    H = cfg.history_len
    valid = _valid_sequence_indices(ep, H, cfg.skip_first_n)
    if len(valid) == 0:
        raise RuntimeError("No valid sequence centers; check history_len / skip_first_n.")
    _, val_idx, _ = _split_by_episode(valid, ep, cfg.val_frac, cfg.test_frac, cfg.split_seed)
    if val_idx is None or len(val_idx) == 0:
        raise RuntimeError("No validation centers (val_frac or split produced empty val).")

    val_eps = ep[val_idx]
    unique_eps = np.unique(val_eps)
    if episode_id is not None:
        if int(episode_id) not in unique_eps:
            raise ValueError(
                f"EPISODE_ID={episode_id} not among validation episodes {sorted(unique_eps.tolist())[:20]}..."
            )
        best_ep = int(episode_id)
    else:
        dataset_mid = len(ep) // 2
        best_ep, best_dist = int(unique_eps[0]), float("inf")
        for ue in unique_eps:
            ks = val_idx[val_eps == ue]
            d = abs(int(np.median(ks)) - dataset_mid)
            if d < best_dist:
                best_dist, best_ep = d, int(ue)

    centers = np.sort(val_idx[val_eps == best_ep])
    if len(centers) == 0:
        raise RuntimeError(f"No validation sequence centers in episode {best_ep}.")

    vae = BlockVAE(cfg, rngs=nnx.Rngs(params=rng))
    vae_path = cfg.vae_ckpt or os.path.join(cfg.ckpt_dir, "vae")
    if os.path.isdir(vae_path):
        _load_latest_ckpt(vae, vae_path)
    else:
        _load_ckpt(vae, vae_path)

    use_visual = cfg.use_visual
    rng, pk = jax.random.split(rng)
    if use_visual:
        predictor = HistoryPredictorWithVision(cfg, rngs=nnx.Rngs(params=pk))
    else:
        predictor = HistoryPredictor(cfg, rngs=nnx.Rngs(params=pk))
    step_loaded = _load_dynamics_ckpt(predictor, dyn_ckpt)
    if use_visual:
        linen_params = _download_resnet10_params()
        _map_linen_to_nnx_backbone(linen_params, predictor.vis_encoder.backbone)
    logging.info("Stage3 vis: VAE loaded | dynamics (%s) loaded step=%s from %s",
                 type(predictor).__name__, step_loaded, dyn_ckpt)

    sub = "dynamics_vis" if use_visual else "dynamics"
    out_root = os.path.join(cfg.ckpt_dir, sub, f"episode_pred_vis_ep{best_ep}")
    os.makedirs(out_root, exist_ok=True)

    maes_l, maes_r = [], []
    for fi, k in enumerate(centers):
        frames = [base_ds[int(k - H + 1 + i)] for i in range(H + 1)]
        tl_h = np.stack([f["tactile_left"] for f in frames[:H]], axis=0)
        tr_h = np.stack([f["tactile_right"] for f in frames[:H]], axis=0)
        ah = np.stack([f["action"] for f in frames[:H]], axis=0)
        tgt_l = np.asarray(frames[H]["tactile_left"], dtype=np.float32)
        tgt_r = np.asarray(frames[H]["tactile_right"], dtype=np.float32)

        tl_b = jnp.expand_dims(jnp.asarray(tl_h), 0)
        tr_b = jnp.expand_dims(jnp.asarray(tr_h), 0)
        a_b = jnp.expand_dims(jnp.asarray(ah), 0)

        if use_visual:
            wh = np.stack([f["wrist_image"] for f in frames[:H]], axis=0)
            sh = np.stack([f["state"] for f in frames[:H]], axis=0)
            w_b = jnp.expand_dims(jnp.asarray(wh), 0)
            s_b = jnp.expand_dims(jnp.asarray(sh), 0)
            pl, pr = _jit_dynamics_predict_pixels_vis(vae, predictor, tl_b, tr_b, a_b, w_b, s_b)
        else:
            pl, pr = _jit_dynamics_predict_pixels(vae, predictor, tl_b, tr_b, a_b)
        pl = np.asarray(jax.device_get(pl[0]))
        pr = np.asarray(jax.device_get(pr[0]))

        el, er = np.abs(tgt_l - pl), np.abs(tgt_r - pr)
        maes_l.append(float(np.mean(el)))
        maes_r.append(float(np.mean(er)))

        fig, axes = plt.subplots(6, 5, figsize=(15, 18))
        for t in range(5):
            vmin_l = float(min(tgt_l[t].min(), pl[t].min()))
            vmax_l = float(max(tgt_l[t].max(), pl[t].max()))
            axes[0, t].imshow(tgt_l[t], cmap="hot", vmin=vmin_l, vmax=vmax_l, aspect="equal")
            axes[0, t].set_title(f"L GT t={t}", fontsize=8)
            axes[0, t].axis("off")
            axes[1, t].imshow(pl[t], cmap="hot", vmin=vmin_l, vmax=vmax_l, aspect="equal")
            axes[1, t].set_title(f"L pred", fontsize=8)
            axes[1, t].axis("off")
            emax_l = float(max(el[t].max(), 1e-8))
            im_el = axes[2, t].imshow(el[t], cmap="hot", vmin=0, vmax=emax_l, aspect="equal")
            axes[2, t].set_title(f"L |err|", fontsize=8)
            axes[2, t].axis("off")

            vmin_r = float(min(tgt_r[t].min(), pr[t].min()))
            vmax_r = float(max(tgt_r[t].max(), pr[t].max()))
            axes[3, t].imshow(tgt_r[t], cmap="hot", vmin=vmin_r, vmax=vmax_r, aspect="equal")
            axes[3, t].set_title(f"R GT t={t}", fontsize=8)
            axes[3, t].axis("off")
            axes[4, t].imshow(pr[t], cmap="hot", vmin=vmin_r, vmax=vmax_r, aspect="equal")
            axes[4, t].set_title(f"R pred", fontsize=8)
            axes[4, t].axis("off")
            emax_r = float(max(er[t].max(), 1e-8))
            im_er = axes[5, t].imshow(er[t], cmap="hot", vmin=0, vmax=emax_r, aspect="equal")
            axes[5, t].set_title(f"R |err|", fontsize=8)
            axes[5, t].axis("off")

        fig.colorbar(im_el, ax=axes[2, :].tolist(), fraction=0.02, pad=0.02)
        fig.colorbar(im_er, ax=axes[5, :].tolist(), fraction=0.02, pad=0.02)
        mae_l, mae_r = maes_l[-1], maes_r[-1]
        fig.suptitle(
            f"ep={best_ep} step={fi}/{len(centers)} center_k={k} target_frame={k + 1} | "
            f"MAE_L={mae_l:.5f} MAE_R={mae_r:.5f}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(os.path.join(out_root, f"pred_frame_{fi:04d}_k{k}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    fig_sum, ax_sum = plt.subplots(figsize=(12, 4))
    ax_sum.plot(maes_l, label="MAE left (next block)", color="tab:red")
    ax_sum.plot(maes_r, label="MAE right (next block)", color="tab:blue", alpha=0.8)
    ax_sum.set_xlabel("Index in episode (valid val centers)")
    ax_sum.set_ylabel("MAE")
    ax_sum.set_title(f"Stage3 dynamics pred | ep={best_ep} | {len(centers)} steps")
    ax_sum.legend()
    ax_sum.grid(True, alpha=0.3)
    fig_sum.tight_layout()
    fig_sum.savefig(os.path.join(out_root, "timeline_mae.png"), dpi=120)
    plt.close(fig_sum)

    logging.info(
        "Stage3 vis done → %s (%d prediction steps). mean MAE L=%.5f R=%.5f",
        out_root,
        len(centers),
        float(np.mean(maes_l)),
        float(np.mean(maes_r)),
    )


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main(train_config: _config.TrainConfig):
    cfg = WMConfig()
    cfg.stage = int(os.environ.get("STAGE", str(cfg.stage)))
    cfg.vae_ckpt = os.environ.get("VAE_CKPT", cfg.vae_ckpt)
    if os.environ.get("HISTORY_LEN"):
        cfg.history_len = int(os.environ["HISTORY_LEN"])
    if os.environ.get("USE_VISUAL", "").strip().lower() in ("1", "true", "yes"):
        cfg.use_visual = True
    if os.environ.get("WRIST_IMAGE_KEY"):
        cfg.wrist_image_key = os.environ["WRIST_IMAGE_KEY"]
    if os.environ.get("STATE_KEY"):
        cfg.state_key = os.environ["STATE_KEY"]
    if os.environ.get("STATE_DIM"):
        cfg.state_dim = int(os.environ["STATE_DIM"])
    dyn_ckpt_env = os.environ.get("DYN_CKPT", "").strip()
    episode_id_env = os.environ.get("EPISODE_ID", "").strip()
    episode_id_parsed: int | None = int(episode_id_env) if episode_id_env else None

    _init_logging()
    logging.info("Platform: %s", platform.node())
    logging.info("Stage: %d | WMConfig: %s", cfg.stage, dataclasses.asdict(cfg))
    logging.info("batch_size=%d  steps=%d  seed=%d", train_config.batch_size, train_config.num_train_steps, train_config.seed)

    rng = jax.random.key(train_config.seed)

    # ── Load dataset ──
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    repo_id = data_config.repo_id
    logging.info("Dataset: %s", repo_id)

    raw_dataset = lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps={"action": [0.0]})

    repack_mapping: dict[str, str] = {
        "observation/tactile_left": "observation.tactile_left",
        "observation/tactile_right": "observation.tactile_right",
        "actions": "action",
        "episode_index": "episode_index",
    }
    if cfg.use_visual:
        repack_mapping["wrist_image"] = cfg.wrist_image_key
        repack_mapping["state"] = cfg.state_key
        logging.info("Visual conditioning enabled: wrist_image_key=%s  state_key=%s  state_dim=%d",
                     cfg.wrist_image_key, cfg.state_key, cfg.state_dim)

    repack = _transforms.RepackTransform(repack_mapping)
    block_tf = BlockTransform(cfg.action_dim, extract_visual=cfg.use_visual)
    base_ds = TransformedDataset(raw_dataset, [repack, block_tf])

    if cfg.use_visual and "STATE_DIM" not in os.environ:
        try:
            s0 = base_ds[0]
            if "state" in s0:
                inferred = int(np.asarray(s0["state"], dtype=np.float32).size)
                if inferred != cfg.state_dim:
                    logging.info(
                        "state_dim: 使用数据集推断值 %d（WMConfig 默认曾为 %d）；"
                        "若不对请设环境变量 STATE_DIM",
                        inferred,
                        cfg.state_dim,
                    )
                cfg.state_dim = inferred
        except Exception as ex:
            logging.warning("无法从首帧推断 state_dim，沿用 WMConfig.state_dim=%d: %s", cfg.state_dim, ex)

    ep_per_frame = _get_frame_episodes(raw_dataset)
    local_batch = train_config.batch_size // jax.process_count()
    logging.info("Frames: %d  local_batch: %d", len(raw_dataset), local_batch)

    if cfg.stage == 1:
        _run_stage1(cfg, train_config, base_ds, ep_per_frame, local_batch, rng)
    elif cfg.stage == 2:
        _run_stage2(cfg, train_config, base_ds, ep_per_frame, local_batch, rng)
    elif cfg.stage == 3:
        if not dyn_ckpt_env:
            raise ValueError(
                "STAGE=3 requires DYN_CKPT=/path/to/dynamics_step.pkl "
                "or DYN_CKPT=/path/to/dynamics_dir (latest .pkl)."
            )
        wandb.init(mode="disabled")
        _run_stage3_episode_pred_vis(
            cfg, base_ds, ep_per_frame, rng,
            dyn_ckpt=dyn_ckpt_env,
            episode_id=episode_id_parsed,
        )
    else:
        raise ValueError(f"STAGE must be 1, 2, or 3, got {cfg.stage}")


def _ensure_trainconfig_exp_name(argv: list[str]) -> None:
    """TrainConfig.exp_name is tyro.MISSING for many presets; inject a default if absent."""
    for a in argv:
        if a == "--exp-name" or a.startswith("--exp-name="):
            return
    default = os.environ.get("EXP_NAME", "tactile_world_model")
    argv.extend(["--exp-name", default])


if __name__ == "__main__":
    _ensure_trainconfig_exp_name(sys.argv)
    main(_config.cli())
