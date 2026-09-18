import dataclasses
import functools
import logging
import platform
import copy
from typing import Any, Optional, SupportsIndex, Sequence, Literal
from flax import struct
from openpi.training.data_loader import TransformedDataset, TorchDataLoader, create_torch_dataset


import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb
import torch

import os
import pickle as pkl
import urllib.request
import orbax.checkpoint as ocp
import flax.linen as nn_linen

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms

import time, logging
def tick(msg):
    logging.info(f"[TICK] {msg} t={time.time():.3f}")

@struct.dataclass
class RLBatch:
    obs: _model.Observation
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    masks: jnp.ndarray
    next_obs: _model.Observation
    tasks: Optional[Any] = None
    intervened: jnp.ndarray | None = None
    mc_returns: jnp.ndarray | None = None
    embeddings: jnp.ndarray | None = None
    next_embeddings: jnp.ndarray | None = None
    # Naming: dataset column ``cost`` = raw scalar; ``costs`` here = pipeline tensor (e.g. binarized c_t).
    costs: jnp.ndarray | None = None
    cost_mc_returns: jnp.ndarray | None = None

@at.typecheck
@struct.dataclass
class RLTrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef  # 这里放你的 ActorCritic graphdef（不是 BaseModel 了）
    
    # target network params（至少包含 critic 的 target）
    target_params: nnx.State

    # three optimizers: actor / reward-critic / cost-critic
    opt_state_actor: optax.OptState
    opt_state_critic: optax.OptState
    opt_state_cost_critic: optax.OptState | None
    tx_actor: optax.GradientTransformation = struct.field(pytree_node=False)
    tx_critic: optax.GradientTransformation = struct.field(pytree_node=False)
    tx_cost_critic: optax.GradientTransformation | None = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None

    actor_filter: Any = struct.field(pytree_node=False, default=None)
    reward_critic_filter: Any = struct.field(pytree_node=False, default=None)
    cost_critic_filter: Any = struct.field(pytree_node=False, default=None)
    all_critic_filter: Any = struct.field(pytree_node=False, default=None)

    lag_lambda: jnp.ndarray | None = None

def _get_scalar(x) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _get_nested(sample: dict, key: str) -> Any:
    """Get value by flat key or dot-separated path (e.g. observation.tactile_left)."""
    if key in sample:
        return sample[key]
    parts = key.split(".", 1)
    if len(parts) == 2:
        head, tail = parts
        sub = sample.get(head)
        if isinstance(sub, dict):
            return sub.get(tail) if "." not in tail else _get_nested(sub, tail)
    return None


def _tactile_max_from_sample(sample: dict, key_left: str, key_right: str) -> float:
    """Per-frame FSR/tactile value = max over tactile_left and tactile_right arrays."""
    out = []
    for key in (key_left, key_right):
        arr = _get_nested(sample, key)
        if arr is None:
            continue
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        arr = np.asarray(arr).flatten()
        if arr.size:
            out.append(float(np.max(arr)))
    return float(np.max(out)) if out else float("nan")


def _as_numpy_float_array(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.stack([np.asarray(v, dtype=np.float32) for v in arr], axis=0)
    return arr.astype(np.float32, copy=False)


def _tactile_sequence_from_sample(sample: dict, key: str) -> np.ndarray | None:
    """Return tactile as a time sequence, supporting (T,H,W), (T,M), or flat arrays."""
    arr = _as_numpy_float_array(_get_nested(sample, key))
    if arr is None or arr.size == 0:
        return None
    arr = np.maximum(arr, 0.0)
    if arr.ndim == 1:
        side = int(round(np.sqrt(arr.size)))
        if side * side == arr.size:
            return arr.reshape(1, side, side)
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        side = int(round(np.sqrt(arr.shape[-1])))
        if side * side == arr.shape[-1]:
            return arr.reshape(arr.shape[0], side, side)
        return arr
    if arr.ndim >= 3:
        return arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
    return arr.reshape(1, -1)


def _latest_tactile_frame(seq: np.ndarray | None) -> np.ndarray | None:
    if seq is None or seq.size == 0:
        return None
    return np.asarray(seq[-1], dtype=np.float32)


def _tactile_force(frame: np.ndarray | None) -> float:
    if frame is None or frame.size == 0:
        return 0.0
    return float(np.sum(np.maximum(frame, 0.0)))


def _tactile_contact_area(frame: np.ndarray | None, threshold: float) -> float:
    if frame is None or frame.size == 0:
        return 0.0
    return float(np.count_nonzero(frame > threshold))


def _center_of_pressure(frame: np.ndarray | None) -> np.ndarray | None:
    if frame is None or frame.size == 0:
        return None
    frame = np.maximum(np.asarray(frame, dtype=np.float32), 0.0)
    total = float(np.sum(frame))
    if total <= 1e-6:
        return None
    if frame.ndim == 1:
        idx = np.arange(frame.shape[0], dtype=np.float32)
        return np.asarray([float(np.sum(idx * frame) / total)], dtype=np.float32)
    yy, xx = np.indices(frame.shape[-2:], dtype=np.float32)
    return np.asarray(
        [
            float(np.sum(xx * frame) / total),
            float(np.sum(yy * frame) / total),
        ],
        dtype=np.float32,
    )


def _tactile_slip_cost(seq: np.ndarray | None) -> float:
    """Squared CoP displacement between the two newest frames in a tactile window."""
    if seq is None or len(seq) < 2:
        return 0.0
    cop_prev = _center_of_pressure(seq[-2])
    cop_curr = _center_of_pressure(seq[-1])
    if cop_prev is None or cop_curr is None:
        return 0.0
    diff = cop_curr - cop_prev
    return float(np.sum(diff * diff))


def _compute_tactile_risk_cost(
    sample: dict,
    *,
    tactile_left_key: str,
    tactile_right_key: str,
    force_max: float,
    asym_delta: float,
    area_min: float,
    contact_threshold: float,
    force_weight: float,
    slip_weight: float,
    asym_weight: float,
    area_weight: float,
) -> tuple[float, dict[str, float], bool]:
    left_seq = _tactile_sequence_from_sample(sample, tactile_left_key)
    right_seq = _tactile_sequence_from_sample(sample, tactile_right_key)
    if left_seq is None and right_seq is None:
        return 0.0, {}, False

    left_frame = _latest_tactile_frame(left_seq)
    right_frame = _latest_tactile_frame(right_seq)
    left_force = _tactile_force(left_frame)
    right_force = _tactile_force(right_frame)
    total_force = left_force + right_force

    force_term = max(0.0, total_force - force_max) ** 2
    slip_term = _tactile_slip_cost(left_seq) + _tactile_slip_cost(right_seq)
    asym_term = max(0.0, abs(left_force - right_force) - asym_delta) ** 2
    area = (
        _tactile_contact_area(left_frame, contact_threshold)
        + _tactile_contact_area(right_frame, contact_threshold)
    )
    area_term = max(0.0, area_min - area) ** 2

    components = {
        "force": force_weight * force_term,
        "slip": slip_weight * slip_term,
        "asym": asym_weight * asym_term,
        "area": area_weight * area_term,
        "left_force": left_force,
        "right_force": right_force,
        "contact_area": area,
    }
    total = components["force"] + components["slip"] + components["asym"] + components["area"]
    return float(total), components, True


class TactileRewardMCReturnWrapper:
    """
    Denser rewards from tactile (tactile_left, tactile_right) + step penalty + MC returns.
    - Step penalty at each timestep (encourage efficient completion).
    - Terminal reward from dataset (or fixed) on last transition.
    - Intermediate bonus when first run of consecutive frames has tactile > threshold (e.g. stable grasp).
    - mc_return[t] = sum_{k>=0} gamma^k * reward[t+k].
    """
    def __init__(
        self,
        base: Any,
        *,
        episode_key: str = "episode_index",
        tactile_left_key: str = "observation.tactile_left",
        tactile_right_key: str = "observation.tactile_right",
        step_penalty: float = -0.01,
        terminal_reward: float = 1.0,
        use_terminal_from_dataset: bool = True,
        tactile_threshold: float = 1.0,
        tactile_consecutive_frames: int = 5,
        tactile_bonus_reward: float = 2.0,
        gamma: float = 0.99,
    ):
        self._base = base
        self._episode_key = episode_key
        self._key_left = tactile_left_key
        self._key_right = tactile_right_key
        self._step_penalty = step_penalty
        self._terminal_reward = terminal_reward
        self._use_terminal_from_dataset = use_terminal_from_dataset
        self._tactile_threshold = tactile_threshold
        self._tactile_consecutive = tactile_consecutive_frames
        self._tactile_bonus = tactile_bonus_reward
        self._gamma = gamma
        self._reward_at: dict[int, float] = {}
        self._mc_return_at: dict[int, float] = {}
        self._build_episode_rewards_and_returns()

    def _build_episode_rewards_and_returns(self) -> None:
        n = len(self._base)
        if n == 0:
            return

        s0 = self._base[0]
        tl = _get_nested(s0, self._key_left)
        tr = _get_nested(s0, self._key_right)
        logging.info(
            "TactileRewardMCReturnWrapper: dataset has %d frames. "
            "tactile_left (%s) found=%s shape=%s, tactile_right (%s) found=%s shape=%s",
            n,
            self._key_left, tl is not None, getattr(tl, "shape", None),
            self._key_right, tr is not None, getattr(tr, "shape", None),
        )
        if tl is None and tr is None:
            logging.warning(
                "Neither %s nor %s found in sample[0] keys: %s. "
                "Tactile bonus will never fire. Check your key names!",
                self._key_left, self._key_right, list(s0.keys())[:20],
            )

        # Single pass: read episode_index, reward, tactile for every frame (with progress)
        ep_ids = np.zeros(n, dtype=np.float64)
        raw_rewards = np.zeros(n, dtype=np.float32)
        tactile_vals = np.full(n, float("nan"), dtype=np.float32)

        logging.info("Scanning %d frames for episode/reward/tactile info ...", n)
        for i in tqdm.tqdm(range(n), desc="TactileReward scan", dynamic_ncols=True):
            s = self._base[i]
            ep_ids[i] = _get_scalar(s.get(self._episode_key, 0))
            if "rewards" in s:
                raw_rewards[i] = _get_scalar(s["rewards"])
            elif "reward" in s:
                raw_rewards[i] = _get_scalar(s["reward"])
            tactile_vals[i] = _tactile_max_from_sample(s, self._key_left, self._key_right)

        # Split into episodes by episode_index changes
        boundaries = np.where(np.diff(ep_ids) != 0)[0] + 1
        ep_starts = np.concatenate([[0], boundaries])
        ep_ends = np.concatenate([boundaries, [n]])

        for start, end in zip(ep_starts, ep_ends):
            T = end - start
            if T == 0:
                continue
            rewards = raw_rewards[start:end].copy()
            rewards += self._step_penalty
            tmax = tactile_vals[start:end]

            if not self._use_terminal_from_dataset:
                rewards[T - 1] = self._terminal_reward + self._step_penalty

            # First run of consecutive frames with tactile > threshold
            bonus_idx: int | None = None
            run = 0
            for t in range(T):
                if not np.isnan(tmax[t]) and tmax[t] >= self._tactile_threshold:
                    run += 1
                    if run >= self._tactile_consecutive:
                        bonus_idx = t
                        break
                else:
                    run = 0
            if bonus_idx is not None:
                rewards[bonus_idx] += self._tactile_bonus

            mc = np.zeros(T, dtype=np.float32)
            mc[T - 1] = rewards[T - 1]
            for t in range(T - 2, -1, -1):
                mc[t] = rewards[t] + self._gamma * mc[t + 1]

            for t in range(T):
                self._reward_at[start + t] = float(rewards[t])
                self._mc_return_at[start + t] = float(mc[t])

        n_episodes = len(ep_starts)
        n_bonus = sum(1 for r in self._reward_at.values() if r > 0 and abs(r) > 1e-6)
        all_rewards = list(self._reward_at.values())
        all_mc = list(self._mc_return_at.values())
        logging.info(
            "TactileRewardMCReturnWrapper built: %d episodes, %d frames, "
            "reward range=[%.4f, %.4f], mc_return range=[%.4f, %.4f], "
            "frames with positive reward=%d",
            n_episodes, len(all_rewards),
            min(all_rewards) if all_rewards else 0, max(all_rewards) if all_rewards else 0,
            min(all_mc) if all_mc else 0, max(all_mc) if all_mc else 0,
            n_bonus,
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: SupportsIndex) -> dict:
        i = idx.__index__()
        out = dict(self._base[i])
        if i in self._reward_at:
            out["rewards"] = torch.tensor(self._reward_at[i], dtype=torch.float32)
        if i in self._mc_return_at:
            out["mc_returns"] = torch.tensor(self._mc_return_at[i], dtype=torch.float32)
        return out


class CostMCReturnWrapper:
    """Compute per-frame safety cost and discounted cost returns.

    The default source is the thesis tactile risk decomposition:

        C = w_f ReLU(f - f_max)^2
          + w_s ||CoP_t - CoP_{t-1}||^2
          + w_a ReLU(|f_L - f_R| - delta)^2
          + w_A ReLU(A_min - A)^2

    A dataset ``cost`` column can still be used for ablations by setting
    ``cost_source="dataset"``.  ``costs`` is the instantaneous scalar consumed by
    the cost critic; ``cost_mc_returns`` is its backward discounted episode sum.
    """

    def __init__(
        self,
        base,
        *,
        cost_key: str = "cost",
        episode_key: str = "episode_index",
        gamma: float = 0.99,
        cost_threshold: float = 0.5,
        cost_source: Literal["tactile", "dataset", "auto"] = "tactile",
        tactile_left_key: str = "observation.tactile_left",
        tactile_right_key: str = "observation.tactile_right",
        force_max: float = 4.0,
        asym_delta: float = 2.0,
        area_min: float = 4.0,
        contact_threshold: float = 0.5,
        force_weight: float = 1.0,
        slip_weight: float = 3.0,
        asym_weight: float = 0.1,
        area_weight: float = 100.0,
    ):
        self._base = base
        self._cost_key = cost_key
        self._episode_key = episode_key
        self._gamma = gamma
        self._cost_threshold = cost_threshold
        self._cost_source = cost_source
        self._tactile_left_key = tactile_left_key
        self._tactile_right_key = tactile_right_key
        self._force_max = force_max
        self._asym_delta = asym_delta
        self._area_min = area_min
        self._contact_threshold = contact_threshold
        self._force_weight = force_weight
        self._slip_weight = slip_weight
        self._asym_weight = asym_weight
        self._area_weight = area_weight
        self._cost_at: dict[int, float] = {}
        self._cost_mc_at: dict[int, float] = {}
        self._component_at: dict[int, dict[str, float]] = {}
        self._build()

    def _dataset_cost(self, sample: dict) -> float:
        c = sample.get(self._cost_key)
        if c is None:
            return 0.0
        return 1.0 if _get_scalar(c) > self._cost_threshold else 0.0

    def _cost_from_sample(self, sample: dict) -> tuple[float, dict[str, float]]:
        if self._cost_source in ("tactile", "auto"):
            cost, components, found = _compute_tactile_risk_cost(
                sample,
                tactile_left_key=self._tactile_left_key,
                tactile_right_key=self._tactile_right_key,
                force_max=self._force_max,
                asym_delta=self._asym_delta,
                area_min=self._area_min,
                contact_threshold=self._contact_threshold,
                force_weight=self._force_weight,
                slip_weight=self._slip_weight,
                asym_weight=self._asym_weight,
                area_weight=self._area_weight,
            )
            if found or self._cost_source == "tactile":
                return cost, components
        cost = self._dataset_cost(sample)
        return cost, {"dataset": cost}

    def _build(self) -> None:
        n = len(self._base)
        if n == 0:
            return
        ep_ids = np.zeros(n, dtype=np.float64)
        costs_arr = np.zeros(n, dtype=np.float32)
        component_sums: dict[str, float] = {}
        logging.info("CostMCReturnWrapper: scanning %d frames ...", n)
        for i in tqdm.tqdm(range(n), desc="CostMC scan", dynamic_ncols=True):
            s = self._base[i]
            ep_ids[i] = _get_scalar(s.get(self._episode_key, 0))
            cost, components = self._cost_from_sample(s)
            costs_arr[i] = cost
            self._component_at[i] = components
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + float(value)

        nonzero = int(np.count_nonzero(costs_arr))
        logging.info(
            "CostMCReturnWrapper: source=%s, instantaneous thesis cost range=[%.4f, %.4f], "
            "mean=%.4f, nonzero=%d/%d",
            self._cost_source,
            float(np.min(costs_arr)), float(np.max(costs_arr)), float(np.mean(costs_arr)),
            nonzero, n,
        )
        if component_sums:
            logging.info(
                "CostMCReturnWrapper component means: %s",
                {k: round(v / n, 6) for k, v in sorted(component_sums.items())},
            )
        if self._cost_source == "dataset":
            logging.info(
                "Dataset cost mode: raw field %r is binarized with threshold %.2f before entering Q_C.",
                self._cost_key,
                self._cost_threshold,
            )
        elif self._cost_source == "auto":
            logging.info(
                "Auto cost mode: tactile formula is used when tactile fields are present; otherwise dataset "
                "field %r is binarized with threshold %.2f.",
                self._cost_key,
                self._cost_threshold,
            )

        boundaries = np.where(np.diff(ep_ids) != 0)[0] + 1
        ep_starts = np.concatenate([[0], boundaries])
        ep_ends = np.concatenate([boundaries, [n]])

        for start, end in zip(ep_starts, ep_ends):
            T = end - start
            if T == 0:
                continue
            c = costs_arr[start:end].copy()
            mc = np.zeros(T, dtype=np.float32)
            mc[T - 1] = c[T - 1]
            for t in range(T - 2, -1, -1):
                mc[t] = c[t] + self._gamma * mc[t + 1]
            for t in range(T):
                self._cost_at[start + t] = float(c[t])
                self._cost_mc_at[start + t] = float(mc[t])

        all_c = list(self._cost_at.values())
        all_mc = list(self._cost_mc_at.values())
        logging.info(
            "CostMCReturnWrapper pipeline: %d episodes, %d frames, "
            "costs range=[%.4f, %.4f], cost_mc_returns range=[%.4f, %.4f], "
            "frames_with_cost>0=%d",
            len(ep_starts), len(all_c),
            min(all_c) if all_c else 0, max(all_c) if all_c else 0,
            min(all_mc) if all_mc else 0, max(all_mc) if all_mc else 0,
            sum(1 for v in all_c if v > 0),
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: SupportsIndex) -> dict:
        i = idx.__index__()
        out = dict(self._base[i])
        out.pop("costs", None)
        if i in self._cost_at:
            out["cost"] = torch.tensor(self._cost_at[i], dtype=torch.float32)
            out["costs"] = torch.tensor(self._cost_at[i], dtype=torch.float32)
        if i in self._cost_mc_at:
            out["cost_mc_returns"] = torch.tensor(self._cost_mc_at[i], dtype=torch.float32)
        return out


def _tensor_or_scalar_float(x: Any) -> float:
    """Scalar for logging: torch tensor, numpy, or python scalar."""
    if x is None:
        return float("nan")
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    if hasattr(x, "item") and not isinstance(x, (bool, int, float)):
        try:
            return float(x.item())
        except Exception:
            return float("nan")
    return float(x)


def log_first_episode_transition_rl_fields(
    trans_ds: Any,
    *,
    base_for_episode: Any | None = None,
    episode_key: str = "episode_index",
) -> None:
    """Log first-episode transitions with the same RL scalars DataLoaderImplRL packs into RLBatch (post-transform)."""
    n = len(trans_ds)
    if n == 0:
        logging.info("First-episode transition RL dump: empty transition dataset.")
        return

    def _ep_at_trans_idx(i: int) -> int:
        s_tr = trans_ds[i]
        if episode_key in s_tr:
            return int(_tensor_or_scalar_float(s_tr.get(episode_key)))
        if base_for_episode is not None and i < len(base_for_episode):
            return int(_tensor_or_scalar_float(base_for_episode[i].get(episode_key, 0)))
        raise KeyError(episode_key)

    ep0 = _ep_at_trans_idx(0)

    logging.info(
        "First episode transitions (post-transform): episode_index=%s "
        "(cost=dataset raw, costs=→RLBatch; columns: trans_idx, frame_idx, rewards, dones, masks, "
        "cost, costs, cost_mc_returns, mc_returns)",
        ep0,
    )
    for i in range(n):
        try:
            ep_i = _ep_at_trans_idx(i)
        except KeyError:
            logging.warning(
                "First-episode transition dump: episode_index missing after transform and no base_for_episode; "
                "skipping."
            )
            return
        if ep_i != ep0:
            break
        s = trans_ds[i]
        r = _tensor_or_scalar_float(s.get("rewards"))
        d = s.get("dones")
        d_b = bool(d.item()) if isinstance(d, torch.Tensor) else bool(d)
        m = _tensor_or_scalar_float(s.get("masks"))
        cost_raw = _tensor_or_scalar_float(s.get("cost"))
        costs_pipe = _tensor_or_scalar_float(s.get("costs"))
        c_mc = _tensor_or_scalar_float(s.get("cost_mc_returns"))
        mc_r = _tensor_or_scalar_float(s.get("mc_returns"))
        fi = s.get("frame_index")
        fi_i = int(_tensor_or_scalar_float(fi)) if fi is not None else i
        logging.info(
            "  %6d  %6d  %14.6f  %5s  %7.4f  %14.6f  %14.6f  %14.6f  %14.6f",
            i,
            fi_i,
            r,
            str(d_b),
            m,
            cost_raw,
            costs_pipe,
            c_mc,
            mc_r,
        )


class LeRobotTransitionDataset:
    """frame sample -> transition sample (adds next_obs, reward, done, mask)."""
    def __init__(self, base, *, episode_key: str | None = "episode_index", is_last_key: str | None = None):
        self._base = base
        self._episode_key = episode_key
        self._is_last_key = is_last_key

    def __len__(self):
        return len(self._base) - 1

    def _is_boundary(self, s: dict, s_next: dict) -> bool:
        # explicit terminal flag if provided
        if self._is_last_key is not None and self._is_last_key in s:
            v = s[self._is_last_key]
            return bool(v.item()) if isinstance(v, torch.Tensor) else bool(v)

        # episode boundary
        if self._episode_key is not None and (self._episode_key in s) and (self._episode_key in s_next):
            a, b = s[self._episode_key], s_next[self._episode_key]
            a = int(a.item()) if isinstance(a, torch.Tensor) else int(a)
            b = int(b.item()) if isinstance(b, torch.Tensor) else int(b)
            return b != a
        return False

    def __getitem__(self, idx: SupportsIndex) -> dict:
        i = idx.__index__()
        s = self._base[i]
        s_next = self._base[i + 1]

        boundary = self._is_boundary(s, s_next)

        # reward: prefer existing rewards/reward
        if "rewards" in s:
            reward = s["rewards"]
        elif "reward" in s:
            reward = s["reward"]
        else:
            reward = torch.tensor(0.0, dtype=torch.float32)

        # done: prefer dataset-provided dones/done, then OR episode boundary
        if "dones" in s:
            done = s["dones"]
        elif "done" in s:
            done = s["done"]
        else:
            done = torch.tensor(False)

        # normalize types: done -> bool tensor scalar; reward -> float32 scalar
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(bool(done))
        done = done.to(torch.bool)

        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(float(reward), dtype=torch.float32)
        reward = reward.to(torch.float32)

        if boundary:
            done = torch.tensor(True, dtype=torch.bool)

        mask = torch.tensor(0.0 if bool(done.item()) else 1.0, dtype=torch.float32)

        # build next sample (dummy if boundary; masked out anyway)
        next_s = s_next if not boundary else s

        # IMPORTANT: your key is "action" not "actions"
        sample = dict(s)
        sample["rewards"] = reward
        sample["dones"] = done
        sample["masks"] = mask
        sample["next"] = dict(next_s)

        # Optional: also remove placeholder fields if they exist
        sample.pop("next.reward", None)
        sample.pop("next.done", None)

        return sample

# --- PATCH START: make offline RL transform picklable (for num_workers > 0) ---

class OfflineRLFullTF:
    """Picklable callable: apply obs_tf to current sample and sample['next'] if present.
    Keep RL keys (rewards/dones/masks/mc_returns) outside obs_tf.

    Preserves ``cost`` (dataset raw) and ``costs`` (pipeline tensor, e.g. binarized) separately.
    """
    def __init__(self, obs_tf):
        self.obs_tf = obs_tf

    def __call__(self, sample: dict) -> dict:
        next_part = sample.get("next", None)

        # --- 1) pull out RL fields we must keep ---
        rewards = sample.get("rewards", None)
        dones   = sample.get("dones", None)
        masks   = sample.get("masks", None)
        mc_returns = sample.get("mc_returns", None)
        cost_raw = sample.get("cost", None)
        costs = sample.get("costs", None)
        cost_mc_returns = sample.get("cost_mc_returns", None)

        _rl_pop_keys = (
            "rewards", "reward", "dones", "done", "masks", "mask",
            "mc_returns", "contact_flags", "next.reward", "next.done",
            "cost", "costs", "cost_mc_returns",
        )

        # --- 2) transform current obs/action/prompt/etc. ---
        sample_wo_next = dict(sample)
        sample_wo_next.pop("next", None)

        for _k in _rl_pop_keys:
            sample_wo_next.pop(_k, None)

        s = dict(self.obs_tf(sample_wo_next))

        # --- 3) put RL keys back ---
        if rewards is not None:
            s["rewards"] = rewards
        if dones is not None:
            s["dones"] = dones
        if masks is not None:
            s["masks"] = masks
        if mc_returns is not None:
            s["mc_returns"] = mc_returns
        if cost_raw is not None:
            s["cost"] = cost_raw
        if costs is not None:
            s["costs"] = costs
        if cost_mc_returns is not None:
            s["cost_mc_returns"] = cost_mc_returns

        # --- 4) transform next part similarly ---
        if next_part is not None:
            next_wo = dict(next_part)
            next_wo.pop("next", None)

            for _k in _rl_pop_keys:
                next_wo.pop(_k, None)

            next_transformed = dict(self.obs_tf(next_wo))
            s["next"] = next_transformed

        return s

def transform_dataset_offline_rl(dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False):
    """Apply the same OpenPI transforms to current obs and next obs."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. Run scripts/compute_norm_stats.py --config-name=<your-config>."
            )
        norm_stats = data_config.norm_stats

    obs_tf = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])

    full_tf = OfflineRLFullTF(obs_tf)
    return TransformedDataset(dataset, [full_tf])

# --- PATCH END ---

def to_jax(x, dtype=jnp.float32):
    # torch -> numpy -> jax
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return jnp.asarray(np.asarray(x), dtype=dtype)

def ensure_horizon(actions, action_horizon: int | None = None):
    # Actions expected shape: [B, ah, ad]
    if actions.ndim == 1:
        actions = actions[None, None, :]          # [1,1,ad]
    elif actions.ndim == 2:
        actions = actions[:, None, :]             # [B,1,ad]
    if action_horizon is not None:
        if actions.shape[-2] != action_horizon:
            raise ValueError(f"Action horizon mismatch: got {actions.shape[-2]}, expected {action_horizon}")
    return actions


class DataLoaderImplRL:
    def __init__(self, data_config, data_loader, *, action_horizon:int):
        self._data_config = data_config
        self._data_loader = data_loader
        self._action_horizon = action_horizon
    
    def data_config(self):
        return self._data_config
    
    def __iter__(self):
        for batch in self._data_loader:
            batch_next = batch["next"]

            batch_curr = dict(batch)
            batch_curr.pop("next", None)

            obs = _model.Observation.from_dict(batch_curr)
            next_obs = _model.Observation.from_dict(batch_next)

            rewards = jnp.asarray(batch_curr["rewards"], dtype=jnp.float32).reshape((-1,1))
            dones = jnp.asarray(batch_curr["dones"], dtype=jnp.float32).reshape((-1,1))
            masks = jnp.asarray(batch_curr["masks"], dtype=jnp.float32).reshape((-1,1))

            # 兼容 repack 后键名为 "action"（如 UF850）或 "actions"
            raw_actions = batch_curr.get("actions", batch_curr.get("action"))
            if raw_actions is None:
                raise KeyError("batch 中需包含 'actions' 或 'action'")
            actions = ensure_horizon(to_jax(raw_actions), action_horizon=self._action_horizon)

            costs_raw = batch_curr.get("costs", None)
            costs = jnp.asarray(costs_raw, dtype=jnp.float32).reshape((-1, 1)) if costs_raw is not None else None
            cost_mc_raw = batch_curr.get("cost_mc_returns", None)
            cost_mc_returns = jnp.asarray(cost_mc_raw, dtype=jnp.float32).reshape((-1, 1)) if cost_mc_raw is not None else None

            yield RLBatch(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                masks=masks,
                next_obs=next_obs,
                tasks=batch_curr.get("tasks", None),
                intervened=batch_curr.get("intervened", None),
                mc_returns=batch_curr.get("mc_returns", None),
                embeddings=batch_curr.get("embeddings", None),
                next_embeddings=batch_curr.get("next_embeddings", None),
                costs=costs,
                cost_mc_returns=cost_mc_returns,
            )

def create_data_loader_offline_rl(
        config,
        *,
        sharding=None,
        shuffle: bool = False,
        num_batches: int | None = None,
        skip_norm_stats: bool = False,
):
    data_config = config.data.create(config.assets_dirs, config.model)
    base = create_torch_dataset(data_config, config.model.action_horizon, config.model)
    if getattr(config, "tactile_reward_enabled", False):
        base = TactileRewardMCReturnWrapper(
            base,
            episode_key="episode_index",
            tactile_left_key=getattr(config, "tactile_left_key", "observation.tactile_left"),
            tactile_right_key=getattr(config, "tactile_right_key", "observation.tactile_right"),
            step_penalty=getattr(config, "step_penalty", -0.01),
            terminal_reward=getattr(config, "terminal_reward", 1.0),
            use_terminal_from_dataset=getattr(config, "use_terminal_from_dataset", True),
            tactile_threshold=getattr(config, "tactile_threshold", 1.0),
            tactile_consecutive_frames=getattr(config, "tactile_consecutive_frames", 5),
            tactile_bonus_reward=getattr(config, "tactile_bonus_reward", 2.0),
            gamma=getattr(config, "discount", 0.99),
        )
        logging.info(
            "Tactile reward + MC return enabled: step_penalty=%s, tactile_threshold=%s, "
            "tactile_consecutive=%s, tactile_bonus=%s, gamma=%s",
            getattr(config, "step_penalty", -0.01),
            getattr(config, "tactile_threshold", 1.0),
            getattr(config, "tactile_consecutive_frames", 5),
            getattr(config, "tactile_bonus_reward", 2.0),
            getattr(config, "discount", 0.99),
        )
    if getattr(config, "use_safety_critic", False):
        cost_thresh = getattr(config, "cost_binarize_threshold", 0.5)
        base = CostMCReturnWrapper(
            base,
            cost_key=getattr(config, "cost_key", "cost"),
            episode_key="episode_index",
            gamma=getattr(config, "discount", 0.99),
            cost_threshold=cost_thresh,
            cost_source=getattr(config, "tactile_cost_source", "tactile"),
            tactile_left_key=getattr(config, "tactile_left_key", "observation.tactile_left"),
            tactile_right_key=getattr(config, "tactile_right_key", "observation.tactile_right"),
            force_max=getattr(config, "tactile_cost_force_max", 4.0),
            asym_delta=getattr(config, "tactile_cost_asym_delta", 2.0),
            area_min=getattr(config, "tactile_cost_area_min", 4.0),
            contact_threshold=getattr(config, "tactile_cost_contact_threshold", 0.5),
            force_weight=getattr(config, "tactile_cost_force_weight", 1.0),
            slip_weight=getattr(config, "tactile_cost_slip_weight", 3.0),
            asym_weight=getattr(config, "tactile_cost_asym_weight", 0.1),
            area_weight=getattr(config, "tactile_cost_area_weight", 100.0),
        )
        logging.info(
            "Cost MC return wrapper enabled: source=%s, discount=%s, "
            "weights(force/slip/asym/area)=(%.3f, %.3f, %.3f, %.3f)",
            getattr(config, "tactile_cost_source", "tactile"),
            getattr(config, "discount", 0.99),
            getattr(config, "tactile_cost_force_weight", 1.0),
            getattr(config, "tactile_cost_slip_weight", 3.0),
            getattr(config, "tactile_cost_asym_weight", 0.1),
            getattr(config, "tactile_cost_area_weight", 100.0),
        )

    trans = LeRobotTransitionDataset(base, episode_key="episode_index", is_last_key=None)
    trans = transform_dataset_offline_rl(trans, data_config, skip_norm_stats=skip_norm_stats)

    if getattr(config, "log_first_episode_reward_cost", True):
        log_first_episode_transition_rl_fields(trans, base_for_episode=base, episode_key="episode_index")

    s = trans[0]
    logging.info(
        "Offline RL dataset sample: keys=%s, action=%s, actions=%s, next=%s, rewards=%s, dones=%s, masks=%s, "
        "cost(raw)=%s, costs(pipeline)=%s",
        list(s.keys())[:25],
        "action" in s,
        "actions" in s,
        "next" in s,
        "rewards" in s,
        "dones" in s,
        "masks" in s,
        "cost" in s,
        "costs" in s,
    )
    if "next" in s and isinstance(s.get("next"), dict):
        logging.info("Sample['next'] keys: %s", list(s["next"].keys())[:10])

    local_batch_size = config.batch_size // jax.process_count()
    torch_loader = TorchDataLoader(
        trans,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework="jax",
    )

    return DataLoaderImplRL(data_config, torch_loader, action_horizon=config.model.action_horizon)


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)
    else:
        h = logging.StreamHandler()
        h.setFormatter(formatter)
        logger.addHandler(h)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config,
    rng,
    state: RLTrainState,
    batch: RLBatch,
):
    batch = batch.replace(rewards=batch.rewards + config.reward_bias)

    ac_online = nnx.merge(state.model_def, state.params)
    ac_online.train()

    ac_target = nnx.merge(state.model_def, state.target_params)
    ac_target.eval()

    # ---- critic grads (all_critic_filter): reward + cost critic jointly ----
    critic_diff = nnx.DiffState(0, state.all_critic_filter)
    (critic_loss, critic_info), all_critic_grads = nnx.value_and_grad(
        lambda m: calql_critic_loss(
            ac_online=m,
            ac_target=ac_target,
            batch=batch,
            rng=rng,
            discount=config.discount,
            critic_ensemble=config.critic_ensemble,
            critic_subsample_size=config.critic_subsample_size,
            cql_n_actions=config.cql_n_actions,
            cql_temp=config.cql_temp,
            cql_alpha=config.cql_alpha,
            cql_action_sample_method=config.cql_action_sample_method,
            cql_clip_diff_min=config.cql_clip_diff_min,
            cql_clip_diff_max=config.cql_clip_diff_max,
            disable_calql=getattr(config, "disable_calql", False),
            target_q_clip=config.target_q_clip,
            use_safety_critic=config.use_safety_critic,
            cost_critic_ensemble=config.critic_cost_ensemble,
            cost_cql_alpha=config.cost_cql_alpha,
            cost_td_weight=config.cost_td_weight,
        ),
        argnums=critic_diff,
        has_aux=True,
    )(ac_online)

    # Split gradients and params by dict key (bypasses nnx.State.filter() sub-filtering bug)
    all_critic_params = nnx.state(ac_online).filter(state.all_critic_filter)
    rc_grads, cc_grads = _split_state_by_cost(all_critic_grads)
    rc_params, cc_params = _split_state_by_cost(all_critic_params)

    rc_updates, new_opt_state_critic = state.tx_critic.update(rc_grads, state.opt_state_critic, rc_params)
    new_rc_params = optax.apply_updates(rc_params, rc_updates)

    rc_grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
        lambda a, x: a + jnp.sum(x ** 2), rc_grads, jnp.array(0.0)))
    rc_update_norm = jnp.sqrt(_tree_sum_sq_arrays(rc_updates))
    rc_param_norm_before = jnp.sqrt(_tree_sum_sq_values(rc_params))
    rc_param_norm_after = jnp.sqrt(_tree_sum_sq_values(new_rc_params))
    rc_param_delta_norm = jnp.sqrt(_tree_sum_sq_delta(rc_params, new_rc_params))
    rc_delta_critic_mlp = _substate_l2_delta(rc_params, new_rc_params, ("critic",))
    rc_delta_critic_enc = _substate_l2_delta(rc_params, new_rc_params, ("critic_encoder",))
    rc_n_param_leaves = jnp.array(float(len(jax.tree_util.tree_leaves(rc_params))), dtype=jnp.float32)

    cc_grad_norm = jnp.array(0.0)
    cc_update_norm = jnp.array(0.0)
    cc_param_norm_before = jnp.array(0.0)
    cc_param_norm_after = jnp.array(0.0)
    cc_param_delta_norm = jnp.array(0.0)
    cc_delta_cost_mlp = jnp.array(0.0)
    cc_delta_cost_enc = jnp.array(0.0)
    cc_n_param_leaves = jnp.array(0.0)

    new_opt_state_cost_critic = state.opt_state_cost_critic
    if state.tx_cost_critic is not None and state.opt_state_cost_critic is not None:
        cc_grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
            lambda a, x: a + jnp.sum(x ** 2), cc_grads, jnp.array(0.0)))
        cc_param_norm_before = jnp.sqrt(_tree_sum_sq_values(cc_params))
        cc_n_param_leaves = jnp.array(float(len(jax.tree_util.tree_leaves(cc_params))), dtype=jnp.float32)

        cc_updates, new_opt_state_cost_critic = state.tx_cost_critic.update(cc_grads, state.opt_state_cost_critic, cc_params)
        new_cc_params = optax.apply_updates(cc_params, cc_updates)

        cc_update_norm = jnp.sqrt(_tree_sum_sq_arrays(cc_updates))
        cc_param_norm_after = jnp.sqrt(_tree_sum_sq_values(new_cc_params))
        cc_param_delta_norm = jnp.sqrt(_tree_sum_sq_delta(cc_params, new_cc_params))
        cc_delta_cost_mlp = _substate_l2_delta(cc_params, new_cc_params, ("cost_critic",))
        cc_delta_cost_enc = _substate_l2_delta(cc_params, new_cc_params, ("cost_critic_encoder",))

        merged_dict = {**new_rc_params.to_pure_dict(), **new_cc_params.to_pure_dict()}
    else:
        merged_dict = new_rc_params.to_pure_dict()

    all_critic_params.replace_by_pure_dict(merged_dict)
    nnx.update(ac_online, all_critic_params)

    # ---- resolve effective cost weight: Lagrangian lambda or fixed cost_weight ----
    lag_lambda = state.lag_lambda
    if config.use_lagrangian:
        effective_cost_w = lag_lambda
    else:
        effective_cost_w = jnp.array(config.cost_weight, dtype=jnp.float32)

    # ---- actor grads (actor_filter) ----
    actor_diff = nnx.DiffState(0, state.actor_filter)
    (actor_loss, actor_info), actor_grads = nnx.value_and_grad(
        lambda m: conrft_actor_loss(
            ac_online=m,
            batch=batch,
            rng=rng,
            bc_weight=config.bc_weight,
            q_weight=config.q_weight,
            lag_lambda=effective_cost_w,
            use_safety_critic=config.use_safety_critic,
        ),
        argnums=actor_diff,
        has_aux=True,
    )(ac_online)

    actor_params = nnx.state(ac_online).filter(state.actor_filter)
    updates, new_opt_state_actor = state.tx_actor.update(actor_grads, state.opt_state_actor, actor_params)
    new_actor_params = optax.apply_updates(actor_params, updates)
    nnx.update(ac_online, new_actor_params)

    new_params = nnx.state(ac_online)

    # ---- Lagrangian dual update (needs cost critic Q_C; skip when use_safety_critic=False) ----
    if config.use_safety_critic:
        rng, dual_rng = jax.random.split(rng)
        dual_actions = ac_online.sample_actions(dual_rng, batch.obs)
        dual_cost_q = ac_online.q_cost(batch.obs, dual_actions, train=False).mean(axis=0).mean()
        violation = jax.lax.stop_gradient(dual_cost_q - config.cost_limit)
        new_lambda = jnp.clip(lag_lambda + config.lag_lambda_lr * violation, 0.0, config.lag_lambda_max)
    else:
        dual_cost_q = jnp.array(0.0, dtype=jnp.float32)
        violation = jnp.array(0.0, dtype=jnp.float32)
        new_lambda = lag_lambda

    # ---- soft update target critic ----
    new_target_params = soft_update_target_critic(
        state.target_params,
        new_params,
        tau=config.soft_target_tau,
        all_critic_filter=state.all_critic_filter,
    )
    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        target_params=new_target_params,
        opt_state_actor=new_opt_state_actor,
        opt_state_critic=new_opt_state_critic,
        opt_state_cost_critic=new_opt_state_cost_critic,
        lag_lambda=new_lambda,
    )

    info = {
        **critic_info, **actor_info,
        "lag_lambda": new_lambda,
        "effective_cost_weight": effective_cost_w,
        "dual_cost_q": dual_cost_q,
        "dual_violation": violation,
        "rc_grad_norm": rc_grad_norm,
        "rc_update_norm": rc_update_norm,
        "rc_param_norm_before": rc_param_norm_before,
        "rc_param_norm_after": rc_param_norm_after,
        "rc_param_delta_norm": rc_param_delta_norm,
        "rc_delta_critic_mlp": rc_delta_critic_mlp,
        "rc_delta_critic_enc": rc_delta_critic_enc,
        "rc_n_param_leaves": rc_n_param_leaves,
        "cc_grad_norm": cc_grad_norm,
        "cc_update_norm": cc_update_norm,
        "cc_param_norm_before": cc_param_norm_before,
        "cc_param_norm_after": cc_param_norm_after,
        "cc_param_delta_norm": cc_param_delta_norm,
        "cc_delta_cost_mlp": cc_delta_cost_mlp,
        "cc_delta_cost_enc": cc_delta_cost_enc,
        "cc_n_param_leaves": cc_n_param_leaves,
    }
    return new_state, info


@at.typecheck
def train_step_critic_only(
    config,
    rng,
    state: RLTrainState,
    batch: RLBatch,
    cql_alpha: jax.Array,  # 传入 JAX 标量 (f32[]) 以支持 JIT，与 call site jnp.array(..., dtype=jnp.float32) 一致
) -> tuple[RLTrainState, dict]:
    """Only update critic + soft update target; no actor update. cql_alpha 可传入以便 warmup 阶段用更小的值。"""
    # 整条训练链路使用加过 bias 的 reward
    batch = batch.replace(rewards=batch.rewards + config.reward_bias)

    ac_online = nnx.merge(state.model_def, state.params)
    ac_online.train()
    ac_target = nnx.merge(state.model_def, state.target_params)
    ac_target.eval()

    critic_diff = nnx.DiffState(0, state.all_critic_filter)
    (critic_loss, critic_info), all_critic_grads = nnx.value_and_grad(
        lambda m: calql_critic_loss(
            ac_online=m,
            ac_target=ac_target,
            batch=batch,
            rng=rng,
            discount=config.discount,
            critic_ensemble=config.critic_ensemble,
            critic_subsample_size=config.critic_subsample_size,
            cql_n_actions=config.cql_n_actions,
            cql_temp=config.cql_temp,
            cql_alpha=cql_alpha,
            cql_action_sample_method=config.cql_action_sample_method,
            cql_clip_diff_min=config.cql_clip_diff_min,
            cql_clip_diff_max=config.cql_clip_diff_max,
            disable_calql=getattr(config, "disable_calql", False),
            target_q_clip=config.target_q_clip,
            use_safety_critic=config.use_safety_critic,
            cost_critic_ensemble=config.critic_cost_ensemble,
            cost_cql_alpha=config.cost_cql_alpha,
            cost_td_weight=config.cost_td_weight,
        ),
        argnums=critic_diff,
        has_aux=True,
    )(ac_online)

    # Split gradients and params by dict key (bypasses nnx.State.filter() sub-filtering bug)
    all_critic_params = nnx.state(ac_online).filter(state.all_critic_filter)
    rc_grads, cc_grads = _split_state_by_cost(all_critic_grads)
    rc_params, cc_params = _split_state_by_cost(all_critic_params)

    rc_updates, new_opt_state_critic = state.tx_critic.update(rc_grads, state.opt_state_critic, rc_params)
    new_rc_params = optax.apply_updates(rc_params, rc_updates)

    rc_grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
        lambda a, x: a + jnp.sum(x ** 2), rc_grads, jnp.array(0.0)))
    rc_update_norm = jnp.sqrt(_tree_sum_sq_arrays(rc_updates))
    rc_param_norm_before = jnp.sqrt(_tree_sum_sq_values(rc_params))
    rc_param_norm_after = jnp.sqrt(_tree_sum_sq_values(new_rc_params))
    rc_param_delta_norm = jnp.sqrt(_tree_sum_sq_delta(rc_params, new_rc_params))
    rc_delta_critic_mlp = _substate_l2_delta(rc_params, new_rc_params, ("critic",))
    rc_delta_critic_enc = _substate_l2_delta(rc_params, new_rc_params, ("critic_encoder",))
    rc_n_param_leaves = jnp.array(float(len(jax.tree_util.tree_leaves(rc_params))), dtype=jnp.float32)

    cc_grad_norm = jnp.array(0.0)
    cc_update_norm = jnp.array(0.0)
    cc_param_norm_before = jnp.array(0.0)
    cc_param_norm_after = jnp.array(0.0)
    cc_param_delta_norm = jnp.array(0.0)
    cc_delta_cost_mlp = jnp.array(0.0)
    cc_delta_cost_enc = jnp.array(0.0)
    cc_n_param_leaves = jnp.array(0.0)

    new_opt_state_cost_critic = state.opt_state_cost_critic
    if state.tx_cost_critic is not None and state.opt_state_cost_critic is not None:
        cc_grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
            lambda a, x: a + jnp.sum(x ** 2), cc_grads, jnp.array(0.0)))
        cc_param_norm_before = jnp.sqrt(_tree_sum_sq_values(cc_params))
        cc_n_param_leaves = jnp.array(float(len(jax.tree_util.tree_leaves(cc_params))), dtype=jnp.float32)

        cc_updates, new_opt_state_cost_critic = state.tx_cost_critic.update(cc_grads, state.opt_state_cost_critic, cc_params)
        new_cc_params = optax.apply_updates(cc_params, cc_updates)

        cc_update_norm = jnp.sqrt(_tree_sum_sq_arrays(cc_updates))
        cc_param_norm_after = jnp.sqrt(_tree_sum_sq_values(new_cc_params))
        cc_param_delta_norm = jnp.sqrt(_tree_sum_sq_delta(cc_params, new_cc_params))
        cc_delta_cost_mlp = _substate_l2_delta(cc_params, new_cc_params, ("cost_critic",))
        cc_delta_cost_enc = _substate_l2_delta(cc_params, new_cc_params, ("cost_critic_encoder",))

        merged_dict = {**new_rc_params.to_pure_dict(), **new_cc_params.to_pure_dict()}
    else:
        merged_dict = new_rc_params.to_pure_dict()

    all_critic_params.replace_by_pure_dict(merged_dict)
    nnx.update(ac_online, all_critic_params)

    new_params = nnx.state(ac_online)

    new_target_params = soft_update_target_critic(
        state.target_params,
        new_params,
        tau=config.soft_target_tau,
        all_critic_filter=state.all_critic_filter,
    )
    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        target_params=new_target_params,
        opt_state_critic=new_opt_state_critic,
        opt_state_cost_critic=new_opt_state_cost_critic,
    )
    diag = {
        **critic_info,
        "lag_lambda": state.lag_lambda,
        "rc_grad_norm": rc_grad_norm,
        "rc_update_norm": rc_update_norm,
        "rc_param_norm_before": rc_param_norm_before,
        "rc_param_norm_after": rc_param_norm_after,
        "rc_param_delta_norm": rc_param_delta_norm,
        "rc_delta_critic_mlp": rc_delta_critic_mlp,
        "rc_delta_critic_enc": rc_delta_critic_enc,
        "rc_n_param_leaves": rc_n_param_leaves,
        "cc_grad_norm": cc_grad_norm,
        "cc_update_norm": cc_update_norm,
        "cc_param_norm_before": cc_param_norm_before,
        "cc_param_norm_after": cc_param_norm_after,
        "cc_param_delta_norm": cc_param_delta_norm,
        "cc_delta_cost_mlp": cc_delta_cost_mlp,
        "cc_delta_cost_enc": cc_delta_cost_enc,
        "cc_n_param_leaves": cc_n_param_leaves,
    }
    return new_state, diag

# ==========================
# nnx parameter filters
# ==========================

def _split_state_by_cost(state: nnx.State) -> tuple[nnx.State, nnx.State]:
    """Split an nnx.State into (reward_critic_part, cost_critic_part) by top-level key.

    Bypasses nnx.State.filter() which can silently fail on sub-filtered states
    due to VariableState type-checking issues in the NNX filter predicate system.

    Use ``raw_mapping.items()`` instead of ``state[k]``: ``__getitem__`` can raise when a
    child is already ``State`` (see ``_substate_l2_delta`` docstring).
    """
    rc_dict = {}
    cc_dict = {}
    for k, v in state.raw_mapping.items():
        if "cost_critic" in str(k):
            cc_dict[k] = v
        else:
            rc_dict[k] = v
    return nnx.State(rc_dict), nnx.State(cc_dict)


def _tree_sum_sq_values(pytree) -> jax.Array:
    """Sum of squared elements over all leaves (VariableState uses .value)."""

    def _leaf_sq(x):
        v = x.value if hasattr(x, "value") else x
        return jnp.sum(jnp.square(jnp.asarray(v, dtype=jnp.float32)))

    sq = jax.tree_util.tree_map(_leaf_sq, pytree)
    return jax.tree_util.tree_reduce(lambda a, b: a + b, sq, initializer=jnp.array(0.0, dtype=jnp.float32))


def _tree_sum_sq_delta(before, after) -> jax.Array:
    """Sum of squared (after - before) over matching leaves."""

    def _leaf_sq_delta(b, a):
        vb = b.value if hasattr(b, "value") else b
        va = a.value if hasattr(a, "value") else a
        d = jnp.asarray(va, dtype=jnp.float32) - jnp.asarray(vb, dtype=jnp.float32)
        return jnp.sum(jnp.square(d))

    sq = jax.tree_util.tree_map(_leaf_sq_delta, before, after)
    return jax.tree_util.tree_reduce(lambda a, b: a + b, sq, initializer=jnp.array(0.0, dtype=jnp.float32))


def _tree_sum_sq_arrays(pytree) -> jax.Array:
    """Sum of squares for a tree of plain arrays (e.g. grads, optax updates)."""

    def _sq(x):
        return jnp.sum(jnp.square(jnp.asarray(x, dtype=jnp.float32)))

    sq = jax.tree_util.tree_map(_sq, pytree)
    return jax.tree_util.tree_reduce(lambda a, b: a + b, sq, initializer=jnp.array(0.0, dtype=jnp.float32))


def _substate_l2_delta(before: nnx.State, after: nnx.State, names: tuple[str, ...]) -> jax.Array:
    """L2 norm of param delta for leaves under top-level keys in ``names``.

    Uses ``raw_mapping`` + ``jax.tree_util.tree_map`` instead of ``flat_state()``
    path matching, which can silently break after ``optax.apply_updates`` because
    the NNX pytree unflatten may produce different ``flat_state()`` path keys.
    """
    def _leaf_sq_delta(b, a):
        vb = b.value if hasattr(b, "value") else b
        va = a.value if hasattr(a, "value") else a
        d = jnp.asarray(va, dtype=jnp.float32) - jnp.asarray(vb, dtype=jnp.float32)
        return jnp.sum(jnp.square(d))

    total_sq = jnp.array(0.0, dtype=jnp.float32)
    b_map = before.raw_mapping
    a_map = after.raw_mapping
    for name in names:
        if name not in b_map or name not in a_map:
            continue
        sq = jax.tree_util.tree_map(_leaf_sq_delta, b_map[name], a_map[name])
        total_sq = total_sq + jax.tree_util.tree_reduce(
            lambda a, b: a + b, sq, initializer=jnp.array(0.0, dtype=jnp.float32))
    return jnp.sqrt(total_sq)


def make_rl_filters(config):
    """RL 中使用的参数过滤器。

    CRITICAL BUG FIX: nnx.PathContains 做的是路径 *元组* 的精确元素匹配
    （"critic" in ("cost_critic", ...) → False），不是子串匹配。
    所以 PathContains("critic") 匹配不到 "cost_critic" / "critic_encoder" /
    "cost_critic_encoder" 这些 key。

    改用 nnx_utils.PathRegex 做子串匹配，它把路径 join 成 "/" 分隔字符串
    再做正则匹配，可以正确覆盖所有 critic 相关参数。

    reward_critic_filter / cost_critic_filter 已由 _split_state_by_cost()
    基于 dict key 实现，此处仅保留以兼容 RLTrainState 字段。
    """
    base_trainable = config.trainable_filter

    actor_filter = nnx.All(nnx.PathContains("actor"), base_trainable)
    all_critic_filter = nnx.All(nnx_utils.PathRegex(".*critic.*"), base_trainable)

    reward_critic_filter = None
    cost_critic_filter = None

    return actor_filter, reward_critic_filter, cost_critic_filter, all_critic_filter

# ================================================================================================================
# soft_update_target_critic
# ================================================================================================================

def soft_update_target_critic(target_params: nnx.State, online_params: nnx.State, *, tau: float, all_critic_filter) -> nnx.State:
    tgt_c = target_params.filter(all_critic_filter)
    tgt_o = target_params.filter(nnx.Not(all_critic_filter))
    onl_c = online_params.filter(all_critic_filter)

    new_tgt_c = jax.tree_util.tree_map(lambda t, o: (1.0 - tau) * t + tau * o, tgt_c, onl_c)

    # 关键：不要用返回值接 replace_by_pure_dict（它可能返回 None）
    new_target = copy.deepcopy(target_params)   # 或者 target_params.copy() 如果有
    new_target.replace_by_pure_dict({
        **tgt_o.to_pure_dict(),
        **new_tgt_c.to_pure_dict(),
    })
    return new_target

# ================================================================================================================
# ConRFT-CalQL modules (nnx)
# ================================================================================================================
ActionMode = Literal["first", "flatten"]


def action_for_critic(actions: jnp.ndarray, mode: ActionMode) -> jnp.ndarray:
    """
    actions: [B, AH, AD]
    return:
      - "first":   [B, AD]
      - "flatten": [B, AH*AD]
    """
    if mode == "first":
        return actions[:, 0, :]
    elif mode == "flatten":
        b, ah, ad = actions.shape
        return actions.reshape(b, ah * ad)
    else:
        raise ValueError(f"Unknown action mode: {mode}")


class CriticEnsemble(nnx.Module):
    """Q(s,a) ensemble: outputs [E,B]. 这里 state_feat 由外部 encoder 提供。"""
    def __init__(self, in_dim: int, hidden: Sequence[int], ensemble: int, *, rngs: nnx.Rngs):
        self.ensemble = ensemble
        self.mlps = []
        for _ in range(ensemble):
            layers = []
            d = in_dim
            for h in hidden:
                layers += [nnx.Linear(d, h, rngs=rngs), nnx.relu]
                d = h
            layers += [nnx.Linear(d, 1, rngs=rngs)]
            self.mlps.append(nnx.Sequential(*layers))

    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        # x: [B, in_dim]
        qs = [mlp(x)[:, 0] for mlp in self.mlps]  # list of [B]
        return jnp.stack(qs, axis=0)              # [E,B]


# ================================================================================================================
# ConRFT-faithful ResNet-10 critic encoder（ImageNet-1K 预训练冻结 backbone + 可训练 pooling head）
# ================================================================================================================

class GroupNormNNX(nnx.Module):
    """GroupNorm (NNX). Handles both [B,H,W,C] and [H,W,C] inputs."""
    def __init__(self, num_features: int, num_groups: int = 4, epsilon: float = 1e-5, *, rngs: nnx.Rngs):
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.scale = nnx.Param(jnp.ones((num_features,)))
        self.bias = nnx.Param(jnp.zeros((num_features,)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        no_batch = x.ndim == 3
        if no_batch:
            x = x[jnp.newaxis]
        B = x.shape[0]
        spatial = x.shape[1:-1]
        C = x.shape[-1]
        G = self.num_groups
        x = x.reshape(B, *spatial, G, C // G)
        reduce_axes = tuple(range(1, 1 + len(spatial))) + (-1,)
        mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
        var = jnp.var(x, axis=reduce_axes, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.epsilon)
        x = x.reshape(B, *spatial, C)
        x = x * self.scale.value + self.bias.value
        if no_batch:
            x = x[0]
        return x


class ResNetBlockNNX(nnx.Module):
    """ResNet basic block: two 3x3 convs + GroupNorm + residual."""
    def __init__(self, in_filters: int, filters: int, strides: tuple = (1, 1), *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_filters, filters, kernel_size=(3, 3), strides=strides, padding="SAME", use_bias=False, rngs=rngs)
        self.norm1 = GroupNormNNX(filters, num_groups=4, rngs=rngs)
        self.conv2 = nnx.Conv(filters, filters, kernel_size=(3, 3), strides=(1, 1), padding="SAME", use_bias=False, rngs=rngs)
        self.norm2 = GroupNormNNX(filters, num_groups=4, rngs=rngs)

        self.needs_proj = (in_filters != filters) or (strides not in ((1, 1), 1))
        if self.needs_proj:
            self.conv_proj = nnx.Conv(in_filters, filters, kernel_size=(1, 1), strides=strides, padding="SAME", use_bias=False, rngs=rngs)
            self.norm_proj = GroupNormNNX(filters, num_groups=4, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        y = nnx.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.needs_proj:
            residual = self.norm_proj(self.conv_proj(residual))
        return nnx.relu(residual + y)


class SpatialLearnedEmbeddingsNNX(nnx.Module):
    """ConRFT spatial learned embeddings: [B,H,W,C] -> [B, C*num_features]."""
    def __init__(self, height: int, width: int, channel: int, num_features: int = 8, *, rngs: nnx.Rngs):
        self.kernel = nnx.Param(
            jax.nn.initializers.lecun_normal()(rngs.params(), (height, width, channel, num_features))
        )

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        no_batch = features.ndim < 4
        if no_batch:
            features = features[jnp.newaxis]
        B = features.shape[0]
        x = jnp.sum(
            jnp.expand_dims(features, -1) * jnp.expand_dims(self.kernel.value, 0),
            axis=(1, 2),
        )
        x = x.reshape(B, -1)
        if no_batch:
            x = x[0]
        return x


class ResNet10Backbone(nnx.Module):
    """ResNet-10 backbone (stage_sizes=(1,1,1,1), frozen via stop_gradient).

    Matches ConRFT ``resnetv1-10-frozen``: 7x7 stem -> 4 stages of 1 ResNetBlock each,
    ImageNet mean/std normalization, resize to ``image_size``, ``stop_gradient`` on output.
    """
    def __init__(self, num_filters: int = 64, image_size: tuple = (128, 128), *, rngs: nnx.Rngs):
        self.image_size = image_size
        self.num_filters = num_filters
        f = num_filters
        self.conv_init = nnx.Conv(3, f, kernel_size=(7, 7), strides=(2, 2),
                                  padding=((3, 3), (3, 3)), use_bias=False, rngs=rngs)
        self.norm_init = GroupNormNNX(f, num_groups=4, rngs=rngs)
        self.block0 = ResNetBlockNNX(f, f, strides=(1, 1), rngs=rngs)
        self.block1 = ResNetBlockNNX(f, f * 2, strides=(2, 2), rngs=rngs)
        self.block2 = ResNetBlockNNX(f * 2, f * 4, strides=(2, 2), rngs=rngs)
        self.block3 = ResNetBlockNNX(f * 4, f * 8, strides=(2, 2), rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        if x.shape[-3:-1] != self.image_size:
            x = jax.image.resize(x, (*x.shape[:-3], *self.image_size, x.shape[-1]), method="bilinear")
        # OpenPI images are in [-1, 1]; convert to [0, 1] then apply ImageNet normalization
        x = x * 0.5 + 0.5
        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])
        x = (x - mean) / std
        x = nnx.relu(self.norm_init(self.conv_init(x)))
        x = nn_linen.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return jax.lax.stop_gradient(x)


class PreTrainedResNetImageEncoder(nnx.Module):
    """ConRFT-style per-image encoder: frozen ResNet-10 + trainable SpatialLearnedEmbeddings + bottleneck.

    For image_size=(128,128), num_filters=64: backbone -> [B,4,4,512],
    spatial embed -> [B,4096], bottleneck -> [B, out_dim].
    """
    def __init__(
        self,
        out_dim: int = 256,
        num_spatial_blocks: int = 8,
        *,
        num_filters: int = 64,
        image_size: tuple = (128, 128),
        rngs: nnx.Rngs,
    ):
        self.backbone = ResNet10Backbone(num_filters=num_filters, image_size=image_size, rngs=rngs)
        fmap_h = image_size[0] // 32
        fmap_w = image_size[1] // 32
        fmap_c = num_filters * 8
        self.spatial_embed = SpatialLearnedEmbeddingsNNX(fmap_h, fmap_w, fmap_c, num_spatial_blocks, rngs=rngs)
        embed_dim = fmap_c * num_spatial_blocks
        self.bottleneck = nnx.Linear(embed_dim, out_dim, rngs=rngs)
        self.bottleneck_ln = nnx.LayerNorm(out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        x = self.backbone(x, train=train)
        x = self.spatial_embed(x)
        x = self.bottleneck(x)
        x = self.bottleneck_ln(x)
        x = nnx.tanh(x)
        return x


class SingleTactileCNN(nnx.Module):
    """Encode a single-hand tactile window [B, T, H, W] -> [B, out_dim].

    Per-frame 2D CNN with spatial global pool, then temporal max pool, then linear proj.
    freeze_backbone stops gradients before proj.
    """
    def __init__(
        self,
        out_dim: int,
        num_filters: Sequence[int] = (32, 64, 128),
        *,
        freeze_backbone: bool = False,
        rngs: nnx.Rngs,
    ):
        self.out_dim = out_dim
        self.num_filters = tuple(num_filters)
        self.freeze_backbone = freeze_backbone
        self.convs = []
        in_f = 1
        for i, f in enumerate(self.num_filters):
            self.convs.append(
                nnx.Conv(in_f, f, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
            )
            in_f = f
        self.proj = nnx.Linear(self.num_filters[-1], out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        # x: [B, T, H, W]
        B, T, H, W = x.shape
        x = x.reshape(B * T, H, W, 1)
        for conv in self.convs:
            x = nnx.relu(conv(x))
        x = jnp.mean(x, axis=(1, 2))          # spatial global pool -> [B*T, C]
        if self.freeze_backbone:
            x = jax.lax.stop_gradient(x)
        x = x.reshape(B, T, -1)               # [B, T, C]
        x = jnp.max(x, axis=1)                # temporal max pool -> [B, C]
        return self.proj(x)                    # [B, out_dim]


# ---- Pretrained VAE tactile encoder (matches train_tactile_world_model.py) ----

class GRUCellNNX(nnx.Module):
    """Standard GRU cell (matches GRUCell in train_tactile_world_model.py for weight compat)."""
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


class TactileBlockVAEEncoder(nnx.Module):
    """Block-VAE encoder: [B,5,16,16] -> (mu, logvar) each [B, latent_dim].

    Architecture and attribute names identical to ``BlockVAEEncoder`` in
    ``train_tactile_world_model.py`` so that checkpoint weights can be loaded
    directly via ``replace_by_pure_dict``.
    """
    def __init__(
        self,
        latent_dim: int = 64,
        enc_channels: tuple = (32, 64, 128),
        enc_gru_hidden: int = 128,
        tactile_H: int = 16,
        tactile_W: int = 16,
        *,
        rngs: nnx.Rngs,
    ):
        chs = enc_channels
        self.c1 = nnx.Conv(1, chs[0], (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c2 = nnx.Conv(chs[0], chs[1], (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        self.c3 = nnx.Conv(chs[1], chs[2], (3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        cnn_flat = chs[2] * (tactile_H // 8) * (tactile_W // 8)
        self.gru_cell = GRUCellNNX(cnn_flat, enc_gru_hidden, rngs=rngs)
        self.gru_hidden = enc_gru_hidden
        self.fc_mu = nnx.Linear(enc_gru_hidden, latent_dim, rngs=rngs)
        self.fc_logvar = nnx.Linear(enc_gru_hidden, latent_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        B, T, H, W = x.shape
        z = x.reshape(B * T, H, W, 1)
        z = nnx.relu(self.c1(z))
        z = nnx.relu(self.c2(z))
        z = nnx.relu(self.c3(z))
        z = z.reshape(B, T, -1)
        z_seq = jnp.transpose(z, (1, 0, 2))
        h0 = jnp.zeros((B, self.gru_hidden))

        def step(h, x_t):
            return self.gru_cell(x_t, h), None

        h_final, _ = jax.lax.scan(step, h0, z_seq)
        return self.fc_mu(h_final), self.fc_logvar(h_final)


class TactileBlockVAEDecoder(nnx.Module):
    """Block-VAE decoder (mirrors BlockVAEDecoder for checkpoint loading only)."""
    def __init__(self, latent_dim: int = 64, dec_init_spatial: int = 4,
                 dec_init_ch: int = 64, tactile_T: int = 5, *, rngs: nnx.Rngs):
        sp, ch, T = dec_init_spatial, dec_init_ch, tactile_T
        self.T, self.sp, self.ch = T, sp, ch
        self.fc = nnx.Linear(latent_dim, T * sp * sp * ch, rngs=rngs)
        self.up1 = nnx.ConvTranspose(ch, ch // 2, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.up2 = nnx.ConvTranspose(ch // 2, ch // 4, (4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
        self.head = nnx.Conv(ch // 4, 1, (3, 3), padding="SAME", rngs=rngs)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        B = z.shape[0]
        x = nnx.relu(self.fc(z))
        x = x.reshape(B * self.T, self.sp, self.sp, self.ch)
        x = nnx.relu(self.up1(x))
        x = nnx.relu(self.up2(x))
        x = self.head(x)
        return x.reshape(B, self.T, 16, 16)


class TactileBlockVAE(nnx.Module):
    """Full BlockVAE replica (encoder + decoder) for loading Stage-1 checkpoints.

    Structure and attribute names match ``BlockVAE`` in train_tactile_world_model.py
    so that ``replace_by_pure_dict`` works directly.
    """
    def __init__(self, latent_dim: int = 64, *, rngs: nnx.Rngs):
        keys = jax.random.split(rngs.params(), 2)
        self.encoder = TactileBlockVAEEncoder(latent_dim=latent_dim, rngs=nnx.Rngs(params=keys[0]))
        self.decoder = TactileBlockVAEDecoder(latent_dim=latent_dim, rngs=nnx.Rngs(params=keys[1]))


class PreTrainedVAETactileEncoder(nnx.Module):
    """Frozen pretrained VAE encoder for tactile: [B, 5, 16, 16] -> [B, out_dim].

    Uses **mu** (deterministic posterior mean, no sampling noise) as the tactile
    feature.  Encoder backbone is frozen via ``stop_gradient``; an optional
    trainable projection adapts the latent dim to the desired ``out_dim``.
    """
    def __init__(self, out_dim: int = 128, latent_dim: int = 64, *, rngs: nnx.Rngs):
        self.encoder = TactileBlockVAEEncoder(latent_dim=latent_dim, rngs=rngs)
        self.latent_dim = latent_dim
        if out_dim != latent_dim:
            self.proj = nnx.Linear(latent_dim, out_dim, rngs=rngs)
        else:
            self.proj = None

    def __call__(self, x: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        mu, _ = self.encoder(x)
        mu = jax.lax.stop_gradient(mu)
        if self.proj is not None:
            return self.proj(mu)
        return mu


def _load_vae_encoder_state(vae_ckpt_path: str, latent_dim: int = 64) -> nnx.State:
    """Load a Stage-1 BlockVAE checkpoint and return the **encoder** NNX state.

    Mirrors the exact loading pattern used in ``train_tactile_world_model.py``
    Stage 2: create a full BlockVAE → ``replace_by_pure_dict`` → extract encoder.
    """
    if os.path.isdir(vae_ckpt_path):
        files = sorted(f for f in os.listdir(vae_ckpt_path) if f.endswith(".pkl"))
        if not files:
            raise FileNotFoundError(f"No .pkl checkpoint in {vae_ckpt_path}")
        vae_ckpt_path = os.path.join(vae_ckpt_path, files[-1])
    if not os.path.isfile(vae_ckpt_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt_path}")

    logging.info("Loading VAE tactile checkpoint from %s", vae_ckpt_path)
    with open(vae_ckpt_path, "rb") as f:
        ckpt = pkl.load(f)

    ckpt_top_keys = sorted(ckpt["params"].keys()) if isinstance(ckpt["params"], dict) else "N/A"
    logging.info("VAE ckpt params top-level keys: %s", ckpt_top_keys)

    expected_keys = {"encoder", "decoder"}
    if isinstance(ckpt["params"], dict) and not expected_keys.issubset(ckpt["params"].keys()):
        raise ValueError(
            f"This checkpoint does NOT look like a Stage-1 BlockVAE checkpoint. "
            f"Expected top-level keys to include {expected_keys}, "
            f"but got: {ckpt_top_keys}. "
            f"Please point vae_tactile_ckpt to the Stage-1 VAE checkpoint "
            f"(default path: checkpoints/tactile_wm/vae/step_XXXXXXXX.pkl)."
        )

    rng = jax.random.key(0)
    tmp_vae = TactileBlockVAE(latent_dim=latent_dim, rngs=nnx.Rngs(params=rng))
    ms = nnx.state(tmp_vae)
    ms.replace_by_pure_dict(ckpt["params"])
    nnx.update(tmp_vae, ms)

    enc_state = nnx.state(tmp_vae.encoder)
    n_params = sum(np.asarray(v).size for v in jax.tree_util.tree_leaves(enc_state))
    logging.info("VAE encoder loaded: %.2fK params, trained step=%d", n_params / 1e3, ckpt.get("step", -1))
    return enc_state


class CriticResNetEncoder(nnx.Module):
    """ConRFT-faithful critic encoder.

    Per-image key: frozen pretrained ResNet-10 + trainable SpatialLearnedEmbeddings + bottleneck.
    Proprio: Dense -> LayerNorm -> tanh (matching ConRFT ``EncodingWrapper``).
    Tactile (optional): direct tactile CNN by default, or frozen pretrained VAE
    encoder when explicitly requested.
    """
    def __init__(
        self,
        image_keys: Sequence[str],
        state_dim: int,
        *,
        image_out_dim: int = 256,
        use_proprio: bool = True,
        proprio_latent_dim: int = 64,
        num_spatial_blocks: int = 8,
        use_tactile: bool = False,
        tactile_out_dim: int = 128,
        tactile_encoder_type: str = "cnn",
        tactile_freeze_backbone: bool = False,
        vae_latent_dim: int = 64,
        encoder_base_rng: jax.Array,
        # Image ResNet backbone follows ConRFT-style freezing. Tactile has a
        # separate freeze flag because its direct CNN is randomly initialized.
        freeze_backbone: bool = True,
    ):
        self.image_keys = tuple(image_keys)
        self.state_dim = state_dim
        self.image_out_dim = image_out_dim
        self.use_proprio = use_proprio
        self.proprio_latent_dim = proprio_latent_dim
        self.use_tactile = use_tactile
        self.tactile_out_dim = tactile_out_dim
        self.tactile_encoder_type = tactile_encoder_type

        n_enc = len(self.image_keys)
        n_tactile = 1 if use_tactile else 0
        keys = jax.random.split(encoder_base_rng, n_enc + n_tactile + 1)

        self.encoders = [
            PreTrainedResNetImageEncoder(
                out_dim=image_out_dim,
                num_spatial_blocks=num_spatial_blocks,
                rngs=nnx.Rngs(params=keys[i]),
            )
            for i in range(n_enc)
        ]

        if use_tactile:
            if tactile_encoder_type == "vae":
                self.tactile_enc = PreTrainedVAETactileEncoder(
                    out_dim=tactile_out_dim,
                    latent_dim=vae_latent_dim,
                    rngs=nnx.Rngs(params=keys[n_enc]),
                )
            elif tactile_encoder_type == "cnn":
                self.tactile_enc = SingleTactileCNN(
                    out_dim=tactile_out_dim,
                    freeze_backbone=tactile_freeze_backbone,
                    rngs=nnx.Rngs(params=keys[n_enc]),
                )
            else:
                raise ValueError(f"Unknown tactile_encoder_type={tactile_encoder_type!r}; use 'cnn' or 'vae'.")
        else:
            self.tactile_enc = None

        prop_key_idx = n_enc + n_tactile
        if use_proprio and proprio_latent_dim is not None:
            self.state_proj = nnx.Linear(state_dim, proprio_latent_dim, rngs=nnx.Rngs(params=keys[prop_key_idx]))
            self.state_ln = nnx.LayerNorm(proprio_latent_dim, rngs=nnx.Rngs(params=keys[prop_key_idx]))
            self._state_out_dim = proprio_latent_dim
        else:
            self.state_proj = None
            self.state_ln = None
            self._state_out_dim = state_dim

        self._out_dim = (
            len(self.image_keys) * image_out_dim
            + (2 * tactile_out_dim if use_tactile else 0)
            + self._state_out_dim
        )

    def __call__(self, obs: _model.Observation, *, train: bool = False) -> jnp.ndarray:
        obs_proc = _model.preprocess_observation(None, obs, train=False)
        encoded = []
        for k, enc in zip(self.image_keys, self.encoders):
            img = obs_proc.images[k]  # [B, H, W, C]
            encoded.append(enc(img, train=train))

        if self.use_tactile:
            missing = []
            if obs.tactile_left is None:
                missing.append("tactile_left")
            if obs.tactile_right is None:
                missing.append("tactile_right")
            if missing:
                raise ValueError(
                    "critic_encoder_use_tactile=True but Observation is missing "
                    + ", ".join(missing)
                    + ". Disable critic_encoder_use_tactile or provide tactile fields."
                )
            encoded.append(self.tactile_enc(obs.tactile_left, train=train))
            encoded.append(self.tactile_enc(obs.tactile_right, train=train))

        feat = jnp.concatenate(encoded, axis=-1)

        if self.use_proprio:
            state = obs_proc.state  # [B, state_dim]
            if self.state_proj is not None:
                state = nnx.tanh(self.state_ln(self.state_proj(state)))
            feat = jnp.concatenate([feat, state], axis=-1)
        return feat  # [B, _out_dim]


# ---- ResNet-10 pretrained weight download & loading ----

def _download_resnet10_params() -> dict:
    """Download (or load cached) ImageNet-1K pretrained ResNet-10 params (Linen format)."""
    file_name = "resnet10_params.pkl"
    cache_dir = os.path.expanduser("~/.serl/")
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(file_path):
        url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
        logging.info("Downloading ResNet-10 pretrained weights from %s ...", url)
        urllib.request.urlretrieve(url, file_path)
        logging.info("Download complete -> %s", file_path)
    else:
        logging.info("ResNet-10 pretrained weights found at %s", file_path)

    with open(file_path, "rb") as f:
        params = pkl.load(f)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logging.info("Loaded %.2fM params from ResNet-10 pretrained on ImageNet-1K", param_count / 1e6)
    return params


def _map_linen_to_nnx_backbone(linen_params: dict, backbone: ResNet10Backbone) -> None:
    """Copy Linen ResNet-10 pretrained weights into NNX ``ResNet10Backbone``."""
    backbone.conv_init.kernel.value = jnp.asarray(linen_params["conv_init"]["kernel"])
    backbone.norm_init.scale.value = jnp.asarray(linen_params["norm_init"]["scale"])
    backbone.norm_init.bias.value = jnp.asarray(linen_params["norm_init"]["bias"])

    blocks = [backbone.block0, backbone.block1, backbone.block2, backbone.block3]
    for i, block in enumerate(blocks):
        lb = linen_params[f"ResNetBlock_{i}"]
        block.conv1.kernel.value = jnp.asarray(lb["Conv_0"]["kernel"])
        block.norm1.scale.value = jnp.asarray(lb["MyGroupNorm_0"]["scale"])
        block.norm1.bias.value = jnp.asarray(lb["MyGroupNorm_0"]["bias"])
        block.conv2.kernel.value = jnp.asarray(lb["Conv_1"]["kernel"])
        block.norm2.scale.value = jnp.asarray(lb["MyGroupNorm_1"]["scale"])
        block.norm2.bias.value = jnp.asarray(lb["MyGroupNorm_1"]["bias"])
        if "conv_proj" in lb:
            block.conv_proj.kernel.value = jnp.asarray(lb["conv_proj"]["kernel"])
            block.norm_proj.scale.value = jnp.asarray(lb["norm_proj"]["scale"])
            block.norm_proj.bias.value = jnp.asarray(lb["norm_proj"]["bias"])


def load_resnet10_pretrained_weights(ac_model: "ActorCriticCalQL") -> None:
    """Download ImageNet-1K ResNet-10 weights and inject into every image encoder in the critic."""
    linen_params = _download_resnet10_params()
    for enc in ac_model.critic_encoder.encoders:
        _map_linen_to_nnx_backbone(linen_params, enc.backbone)
    logging.info("Pretrained ResNet-10 weights injected into %d image encoder(s).", len(ac_model.critic_encoder.encoders))


class ActorCriticCalQL(nnx.Module):
    """
    - actor: OpenPI model (pi0.5 action head etc.)
    - critic_encoder: 独立的 ResNet/CNN 编码器，仅用于 critic，不共享 actor 的 encoder（参考 ConRFT）。
    - critic: CriticEnsemble (standard Cal-QL)。
    """
    def __init__(
        self,
        actor: _model.BaseModel,
        critic_encoder: CriticResNetEncoder,
        *,
        critic_hidden=(256, 256),
        critic_ensemble=2,
        critic_action_mode: ActionMode = "first",
        obs_state_dim: int = None,
        action_horizon: int = None,
        action_dim: int = None,
        use_safety_critic: bool = False,
        cost_critic_ensemble: int = 2,
        cost_critic_encoder: CriticResNetEncoder | None = None,
        rngs: nnx.Rngs,
    ):
        self.actor = actor
        self.critic_encoder = critic_encoder
        self.critic_action_mode = critic_action_mode
        self.use_safety_critic = use_safety_critic

        if critic_action_mode == "first":
            critic_ad = action_dim
        else:
            critic_ad = action_horizon * action_dim

        in_dim = obs_state_dim + critic_ad
        self.critic = CriticEnsemble(in_dim, critic_hidden, critic_ensemble, rngs=rngs)

        if use_safety_critic:
            self.cost_critic_encoder = cost_critic_encoder if cost_critic_encoder is not None else critic_encoder
            rng_cost = nnx.Rngs(params=jax.random.fold_in(rngs.params(), 999))
            self.cost_critic = CriticEnsemble(in_dim, critic_hidden, cost_critic_ensemble, rngs=rng_cost)

    def encode_state_for_critic(self, obs: _model.Observation, *, train: bool) -> jnp.ndarray:
        return self.critic_encoder(obs, train=train)

    def sample_actions(self, rng: jax.Array, obs: _model.Observation) -> jnp.ndarray:
        return self.actor.sample_actions(rng, obs)

    def bc_loss(self, rng: jax.Array, obs: _model.Observation, actions: jnp.ndarray) -> jnp.ndarray:
        chunked = self.actor.compute_loss(rng, obs, actions, train=True)
        return jnp.mean(chunked)

    def q(self, obs: _model.Observation, actions: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        """
        obs: Observation, actions: [B, AH, AD]
        return: [E,B]
        """
        a = action_for_critic(actions, self.critic_action_mode)
        state_feat = self.encode_state_for_critic(obs, train=train)
        x = jnp.concatenate([state_feat, a], axis=-1)
        return self.critic(x, train=train)

    def q_from_state(self, state: jnp.ndarray, actions: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        """
        state:   [B, S_feat]
        actions: [B, AH, AD]
        return:  [E, B]
        """
        a = action_for_critic(actions, self.critic_action_mode)
        x = jnp.concatenate([state, a], axis=-1)
        return self.critic(x, train=train)

    def encode_state_for_cost_critic(self, obs: _model.Observation, *, train: bool) -> jnp.ndarray:
        return self.cost_critic_encoder(obs, train=train)

    def q_cost(self, obs: _model.Observation, actions: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        """Cost critic Q_C(s,a) -> [E, B]. Only valid when use_safety_critic=True."""
        a = action_for_critic(actions, self.critic_action_mode)
        state_feat = self.encode_state_for_cost_critic(obs, train=train)
        x = jnp.concatenate([state_feat, a], axis=-1)
        return self.cost_critic(x, train=train)

    def q_cost_from_state(self, state: jnp.ndarray, actions: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        a = action_for_critic(actions, self.critic_action_mode)
        x = jnp.concatenate([state, a], axis=-1)
        return self.cost_critic(x, train=train)

# ================================================================================================================
# ConRFT-CalQL losses (nnx)
# ================================================================================================================
def repeat_sample_actions(ac: ActorCriticCalQL, rng: jax.Array, obs: _model.Observation, n: int) -> jnp.ndarray:
    """
    采样 n 个动作，用于 CQL current/next actions
    return: [B, n, AH, AD]  (我们最后会在 critic 里转成 [B,3n,ADcrit])
    """
    # 用不同 rng 采样 n 次
    keys = jax.random.split(rng, n)  # [n,2]
    # vmap over n
    acts = jax.vmap(lambda k: ac.sample_actions(k, obs))(keys)  # [n,B,AH,AD]
    return jnp.swapaxes(acts, 0, 1)  # [B,n,AH,AD]


def cql_q_diff_calql(
    ac_online: ActorCriticCalQL,
    ac_for_q: ActorCriticCalQL,
    batch,
    rng: jax.Array,
    *,
    cql_n_actions: int,
    cql_temp: float,
    critic_ensemble: int,
    critic_subsample_size: int | None,
    cql_action_sample_method: str,
    cql_clip_diff_min: float,
    cql_clip_diff_max: float,
):
    info = {}
    B = batch.rewards.shape[0]

    rng, qpred_rng = jax.random.split(rng)
    q_pred = ac_for_q.q(batch.obs, batch.actions, train=True)
    assert q_pred.shape[0] == critic_ensemble

    rng, arng = jax.random.split(rng)
    ah = batch.actions.shape[1]
    ad = batch.actions.shape[2]
    if cql_action_sample_method == "uniform":
        a_rand = jax.random.uniform(arng, (B, cql_n_actions, ah, ad), minval=-1.0, maxval=1.0)
    elif cql_action_sample_method == "normal":
        a_rand = jax.random.normal(arng, (B, cql_n_actions, ah, ad))
    else:
        raise NotImplementedError(cql_action_sample_method)

    rng, cur_rng, nxt_rng = jax.random.split(rng, 3)
    a_cur = repeat_sample_actions(ac_online, cur_rng, batch.obs, cql_n_actions)
    a_nxt = repeat_sample_actions(ac_online, nxt_rng, batch.next_obs, cql_n_actions)

    a_all = jnp.concatenate([a_rand, a_cur, a_nxt], axis=1)
    B, K, AH, AD = a_all.shape
    a_all_flat = a_all.reshape(B * K, AH, AD)
    state_feat = ac_for_q.encode_state_for_critic(batch.obs, train=True)
    state_rep = jnp.repeat(state_feat, K, axis=0)
    q_samples = ac_for_q.q_from_state(state_rep, a_all_flat, train=True)
    q_samples = q_samples.reshape(critic_ensemble, B, K)

    info["all_sampled_action_values"] = q_samples.mean()
    info["random_action_values"] = q_samples[:, :, :cql_n_actions].mean()
    info["current_action_values"] = q_samples[:, :, cql_n_actions:2*cql_n_actions].mean()
    info["next_action_values"] = q_samples[:, :, 2*cql_n_actions:].mean()

    # subsample critics if requested
    if critic_subsample_size is not None:
        rng, sub_rng = jax.random.split(rng)
        idx = jax.random.randint(sub_rng, (critic_subsample_size,), 0, critic_ensemble)
        q_samples = q_samples[idx]
        q_pred = q_pred[idx]
        E_eff = critic_subsample_size
    else:
        E_eff = critic_ensemble

    # ---- Cal-QL lower bound clamp using mc_returns ----
    n_actions_for_calql = 3 * cql_n_actions
    if batch.mc_returns is not None:
        mc = jnp.asarray(batch.mc_returns).reshape(B, 1)  # [B,1]
        mc_lb = jnp.repeat(mc, n_actions_for_calql, axis=1)  # [B,3n]
        num_vals = jnp.size(q_samples[:, :, :n_actions_for_calql])
        info["calql_bound_rate"] = jnp.sum(q_samples < mc_lb[None, :, :]) / num_vals
        q_samples = jnp.maximum(q_samples, mc_lb[None, :, :])
    else:
        info["calql_bound_rate"] = jnp.array(0.0, dtype=jnp.float32)

    # concat q_pred as extra action (正统 ConRFT 写法)
    # shape -> [E_eff,B,3n+1]
    q_cat = jnp.concatenate([q_samples, jnp.expand_dims(q_pred, -1)], axis=-1)
    q_cat = q_cat - jnp.log(q_cat.shape[-1]) * cql_temp

    # logsumexp -> [E_eff,B]
    cql_ood = jax.scipy.special.logsumexp(q_cat / cql_temp, axis=-1) * cql_temp
    info["cql_ood_values"] = cql_ood.mean()

    cql_q_diff = cql_ood - q_pred  # [E_eff,B]

    shared = {"state_rep": state_rep, "a_all_flat": a_all_flat, "K": K}
    return cql_q_diff, info, shared


def calql_critic_loss(
    ac_online: ActorCriticCalQL,
    ac_target: ActorCriticCalQL,
    batch,
    rng: jax.Array,
    *,
    discount: float,
    critic_ensemble: int,
    critic_subsample_size: int | None,
    cql_n_actions: int,
    cql_temp: float,
    cql_alpha: float,
    cql_action_sample_method: str,
    cql_clip_diff_min: float,
    cql_clip_diff_max: float,
    disable_calql: bool = False,
    target_q_clip: float | None = None,
    use_safety_critic: bool = False,
    cost_critic_ensemble: int = 2,
    cost_cql_alpha: float = 0.0,
    cost_td_weight: float = 1.0,
):
    """
    Combined reward + cost critic loss.
      reward: td_loss + cql_alpha * cql_loss                (standard Cal-QL)
      cost:   cost_td_loss + cost_cql_alpha * cql
    Cal-QL for cost: OOD Q-values (CQL term) can be clamped from below by
    cost_mc_returns; TD target uses pure Bellman backup (no MC clamp).
    """
    clip_val = target_q_clip if target_q_clip is not None else 15.0

    B = batch.rewards.shape[0]
    r = jnp.asarray(batch.rewards).reshape(B)
    m = jnp.asarray(batch.masks).reshape(B)

    rng, na_rng = jax.random.split(rng)
    next_actions = ac_online.sample_actions(na_rng, batch.next_obs)

    # ---- Reward critic TD ----
    target_next_qs = ac_target.q(batch.next_obs, next_actions, train=False)

    if critic_subsample_size is not None:
        rng, sub_rng = jax.random.split(rng)
        idx = jax.random.randint(sub_rng, (critic_subsample_size,), 0, critic_ensemble)
        target_next_qs = target_next_qs[idx]

    target_next_min = target_next_qs.min(axis=0)
    target_next_min = jnp.clip(target_next_min, -clip_val, clip_val)
    target_q = r + discount * m * target_next_min
    target_q = jnp.clip(target_q, -clip_val, clip_val)

    q_pred = ac_online.q(batch.obs, batch.actions, train=True)
    target_qs = jnp.broadcast_to(target_q[None, :], q_pred.shape)
    td_loss = jnp.mean((q_pred - target_qs) ** 2)

    # ---- Reward critic CQL / Cal-QL ----
    cql_shared = None
    cql_info = {}
    if disable_calql:
        cql_loss = jnp.array(0.0, dtype=jnp.float32)
        cql_diff_mean = jnp.array(0.0, dtype=jnp.float32)
    else:
        cql_q_diff, cql_info, cql_shared = cql_q_diff_calql(
            ac_online=ac_online,
            ac_for_q=ac_online,
            batch=batch,
            rng=rng,
            cql_n_actions=cql_n_actions,
            cql_temp=cql_temp,
            critic_ensemble=critic_ensemble,
            critic_subsample_size=critic_subsample_size,
            cql_action_sample_method=cql_action_sample_method,
            cql_clip_diff_min=cql_clip_diff_min,
            cql_clip_diff_max=cql_clip_diff_max,
        )
        cql_loss = jnp.clip(cql_q_diff, cql_clip_diff_min, cql_clip_diff_max).mean()
        cql_diff_mean = cql_q_diff.mean()
    reward_critic_loss = td_loss + cql_alpha * cql_loss

    info = {
        "critic_loss": reward_critic_loss,
        "td_loss": td_loss,
        "cql_loss": cql_loss,
        "cql_alpha": jnp.asarray(cql_alpha, jnp.float32),
        "cql_diff": cql_diff_mean,
        "disable_calql": jnp.asarray(disable_calql, jnp.float32),
        "predicted_qs": q_pred.mean(),
        "target_qs": target_qs.mean(),
        "rewards": r.mean(),
        "target_q_clip": jnp.array(clip_val, dtype=jnp.float32),
        **cql_info,
    }

    # ---- Cost critic TD + CQL (separate encoder, reuse OOD actions only) ----
    cost_critic_loss = jnp.array(0.0, dtype=jnp.float32)
    if use_safety_critic and batch.costs is not None:
        c = jnp.asarray(batch.costs).reshape(B)
        cost_target_next_qs = ac_target.q_cost(batch.next_obs, next_actions, train=False)
        if critic_subsample_size is not None:
            rng, csub_rng = jax.random.split(rng)
            cidx = jax.random.randint(csub_rng, (critic_subsample_size,), 0, cost_critic_ensemble)
            cost_target_next_qs = cost_target_next_qs[cidx]
        cost_target_next_max = cost_target_next_qs.max(axis=0)
        cost_target_next_max = jnp.clip(cost_target_next_max, -clip_val, clip_val)
        cost_target_q = c + discount * m * cost_target_next_max
        cost_target_q = jnp.clip(cost_target_q, -clip_val, clip_val)

        cost_q_pred = ac_online.q_cost(batch.obs, batch.actions, train=True)
        cost_target_qs = jnp.broadcast_to(cost_target_q[None, :], cost_q_pred.shape)
        cost_td_loss = jnp.mean((cost_q_pred - cost_target_qs) ** 2)

        if disable_calql:
            cost_cql_loss = jnp.array(0.0, dtype=jnp.float32)
            cost_cql_diff = jnp.array(0.0, dtype=jnp.float32)
            cost_cql_ood = jnp.array(0.0, dtype=jnp.float32)
            cost_critic_loss = cost_td_loss
        else:
            # ---- Cost CQL (flipped): reuse OOD actions, re-encode state with cost encoder ----
            # Reward CQL pushes OOD Q_R DOWN (lower bound) → actor won't exploit OOD for reward.
            # Cost CQL pushes OOD Q_C UP (upper bound) → actor avoids OOD as dangerous.
            # Achieved by flipping the sign: diff = Q_data - logsumexp(Q_OOD) <= 0
            cost_state_feat = ac_online.encode_state_for_cost_critic(batch.obs, train=True)
            a_all_flat = cql_shared["a_all_flat"]
            K = cql_shared["K"]
            cost_state_rep = jnp.repeat(cost_state_feat, K, axis=0)
            cost_q_samples = ac_online.q_cost_from_state(cost_state_rep, a_all_flat, train=True)
            cost_q_samples = cost_q_samples.reshape(cost_critic_ensemble, B, K)

            # Cal-QL lower bound for cost OOD Q-values: prevent OOD actions from
            # being estimated as "cheaper" than what the behavior policy actually incurred.
            # Reward Cal-QL uses jnp.minimum (upper bound); cost mirrors with jnp.maximum.
            if batch.cost_mc_returns is not None:
                cost_mc_lb = jnp.asarray(batch.cost_mc_returns).reshape(1, B, 1)
                cost_q_samples = jnp.maximum(cost_q_samples, cost_mc_lb)

            cost_q_cat = jnp.concatenate([cost_q_samples, jnp.expand_dims(cost_q_pred, -1)], axis=-1)
            cost_q_cat = cost_q_cat - jnp.log(cost_q_cat.shape[-1]) * cql_temp
            cost_cql_ood = jax.scipy.special.logsumexp(cost_q_cat / cql_temp, axis=-1) * cql_temp

            # Flipped sign: Q_data - logsumexp <= 0; minimizing drives logsumexp up → OOD Q_C up
            cost_cql_diff = cost_q_pred - cost_cql_ood
            cost_cql_loss = jnp.clip(cost_cql_diff, cql_clip_diff_min, cql_clip_diff_max).mean()

            cost_critic_loss = cost_td_loss + cost_cql_alpha * cost_cql_loss

        _cost_mc_mean = jnp.asarray(batch.cost_mc_returns).reshape(B).mean() if batch.cost_mc_returns is not None else jnp.array(0.0)
        info.update({
            "cost_td_loss": cost_td_loss,
            "cost_cql_loss": cost_cql_loss,
            "cost_cql_diff": cost_cql_diff.mean() if hasattr(cost_cql_diff, "shape") and cost_cql_diff.shape else cost_cql_diff,
            "cost_critic_loss": cost_critic_loss,
            "cost_predicted_qs": cost_q_pred.mean(),
            "cost_target_qs": cost_target_qs.mean(),
            "cost_mc_returns_mean": _cost_mc_mean,
            "cost_cql_ood_values": cost_cql_ood.mean() if hasattr(cost_cql_ood, "shape") and cost_cql_ood.shape else cost_cql_ood,
            "costs_mean": c.mean(),
        })

    total_critic_loss = reward_critic_loss + cost_td_weight * cost_critic_loss
    info["total_critic_loss"] = total_critic_loss
    info["cost_td_weight"] = jnp.array(cost_td_weight, dtype=jnp.float32)
    return total_critic_loss, info

def conrft_actor_loss(
    ac_online: ActorCriticCalQL,
    batch,
    rng: jax.Array,
    *,
    bc_weight: float,
    q_weight: float,
    lag_lambda: jax.Array,
    use_safety_critic: bool = False,
):
    """
    ConRFT actor loss with Lagrangian cost constraint:
      actor_loss = bc_weight * bc + q_weight * (-Q_R) + lag_lambda * Q_C
    lag_lambda is a learned dual variable (updated outside this function).
    """
    rng, bc_rng = jax.random.split(rng)
    bc_per_ex = ac_online.actor.compute_loss(bc_rng, batch.obs, batch.actions, train=True)
    recon_loss = jnp.mean(bc_per_ex)

    rng, sample_rng = jax.random.split(rng, 2)
    new_actions = ac_online.sample_actions(sample_rng, batch.obs)

    # Reward Q: maximize => -Q_R
    q_new_actions = ac_online.q(batch.obs, new_actions, train=True).mean(axis=0)
    q_loss = -q_new_actions.mean()

    # Cost Q: minimize => +Q_C (max over ensemble for pessimism)
    if use_safety_critic:
        q_cost_new = ac_online.q_cost(batch.obs, new_actions, train=True).max(axis=0)
        cost_q_term = q_cost_new.mean()
    else:
        cost_q_term = jnp.array(0.0, dtype=jnp.float32)

    actor_loss = bc_weight * recon_loss + q_weight * q_loss + lag_lambda * cost_q_term

    info = {
        "actor_loss": actor_loss,
        "bc_loss": recon_loss,
        "q_loss": q_new_actions.mean(),
        "q_sampled_mean": q_new_actions.mean(),
        "cost_q_term": cost_q_term,
    }
    return actor_loss, info

# ================================================================================================================
def _build_rl_train_state(config, rng: jax.Array, batch0: RLBatch, loaded_actor_params: at.Params) -> RLTrainState:
    """Pure function: (rng, batch0, loaded_actor_params) -> RLTrainState. Used for eval_shape and JIT init."""
    rng, actor_rng, enc_rng, cost_enc_rng, ac_rng = jax.random.split(rng, 5)

    actor = config.model.create(actor_rng)
    if loaded_actor_params:
        graphdef, state = nnx.split(actor)
        state.replace_by_pure_dict(loaded_actor_params)
        actor = nnx.merge(graphdef, state)

    image_keys = tuple(batch0.obs.images.keys())
    state_dim = int(batch0.obs.state.shape[-1])
    use_tactile = getattr(config, "critic_encoder_use_tactile", False)
    tactile_encoder_type = getattr(config, "critic_tactile_encoder_type", "cnn")
    tactile_freeze_backbone = getattr(config, "critic_tactile_freeze_backbone", False)
    critic_encoder = CriticResNetEncoder(
        image_keys=image_keys,
        state_dim=state_dim,
        image_out_dim=config.critic_encoder_image_dim,
        use_proprio=True,
        proprio_latent_dim=config.critic_encoder_proprio_dim,
        num_spatial_blocks=getattr(config, "critic_encoder_num_spatial_blocks", 8),
        freeze_backbone=config.critic_encoder_freeze_backbone,
        use_tactile=use_tactile,
        tactile_out_dim=getattr(config, "critic_encoder_tactile_dim", 128),
        tactile_encoder_type=tactile_encoder_type,
        tactile_freeze_backbone=tactile_freeze_backbone,
        vae_latent_dim=getattr(config, "vae_tactile_latent_dim", 64),
        encoder_base_rng=enc_rng,
    )

    cost_critic_encoder = None
    if config.use_safety_critic:
        cost_critic_encoder = CriticResNetEncoder(
            image_keys=image_keys,
            state_dim=state_dim,
            image_out_dim=config.critic_encoder_image_dim,
            use_proprio=True,
            proprio_latent_dim=config.critic_encoder_proprio_dim,
            num_spatial_blocks=getattr(config, "critic_encoder_num_spatial_blocks", 8),
            freeze_backbone=config.critic_encoder_freeze_backbone,
            use_tactile=use_tactile,
            tactile_out_dim=getattr(config, "critic_encoder_tactile_dim", 128),
            tactile_encoder_type=tactile_encoder_type,
            tactile_freeze_backbone=tactile_freeze_backbone,
            vae_latent_dim=getattr(config, "vae_tactile_latent_dim", 64),
            encoder_base_rng=cost_enc_rng,
        )

    obs_state_dim = critic_encoder._out_dim
    action_horizon = batch0.actions.shape[1]
    action_dim = batch0.actions.shape[2]

    ac = ActorCriticCalQL(
        actor=actor,
        critic_encoder=critic_encoder,
        critic_hidden=(256, 256),
        critic_ensemble=config.critic_ensemble,
        critic_action_mode=config.critic_action_mode,
        obs_state_dim=obs_state_dim,
        action_horizon=action_horizon,
        action_dim=action_dim,
        use_safety_critic=config.use_safety_critic,
        cost_critic_ensemble=config.critic_cost_ensemble,
        cost_critic_encoder=cost_critic_encoder,
        rngs=nnx.Rngs(params=ac_rng),
    )

    actor_filter, reward_critic_filter, cost_critic_filter, all_critic_filter = make_rl_filters(config)
    params = nnx.state(ac)
    params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
    target_params = copy.deepcopy(params)

    tx_actor = optax.adamw(config.actor_lr)
    tx_critic = optax.adamw(config.critic_lr)
    actor_params = params.filter(actor_filter)
    all_critic_params = params.filter(all_critic_filter)

    reward_critic_params, cost_critic_params = _split_state_by_cost(all_critic_params)
    opt_state_actor = tx_actor.init(actor_params)
    opt_state_critic = tx_critic.init(reward_critic_params)

    cost_critic_lr = getattr(config, "cost_critic_lr", config.critic_lr)
    tx_cost_critic = optax.adamw(cost_critic_lr) if config.use_safety_critic else None
    opt_state_cost_critic = tx_cost_critic.init(cost_critic_params) if tx_cost_critic is not None else None

    _n_all = len(jax.tree_util.tree_leaves(all_critic_params))
    _n_rc = len(jax.tree_util.tree_leaves(reward_critic_params))
    _n_cc = len(jax.tree_util.tree_leaves(cost_critic_params))
    logging.info("Filter diagnostic: all_critic=%d  reward_critic=%d  cost_critic=%d  (keys: %s)",
                 _n_all, _n_rc, _n_cc, list(all_critic_params.keys()) if hasattr(all_critic_params, 'keys') else 'N/A')
    if _n_cc == 0 and config.use_safety_critic:
        logging.error("cost_critic split matched 0 params — cost critic will NOT be trained!")

    lag_lambda_init = getattr(config, "lag_lambda_init", 0.0)

    return RLTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        model_def=nnx.graphdef(ac),
        target_params=target_params,
        opt_state_actor=opt_state_actor,
        opt_state_critic=opt_state_critic,
        opt_state_cost_critic=opt_state_cost_critic,
        tx_actor=tx_actor,
        tx_critic=tx_critic,
        tx_cost_critic=tx_cost_critic,
        ema_decay=config.ema_decay,
        ema_params=None if config.ema_decay is None else params,
        actor_filter=actor_filter,
        reward_critic_filter=reward_critic_filter,
        cost_critic_filter=cost_critic_filter,
        all_critic_filter=all_critic_filter,
        lag_lambda=jnp.array(lag_lambda_init, dtype=jnp.float32),
    )


def init_rl_train_state(
    config,
    init_rng: jax.Array,
    batch0: RLBatch,
    mesh: jax.sharding.Mesh,
    *,
    data_sharding: jax.sharding.Sharding,
    replicated_sharding: jax.sharding.Sharding,
    resume: bool,
) -> tuple[RLTrainState, jax.sharding.Sharding]:
    """Init RLTrainState with FSDP sharding. Always runs full init so resume can merge into real state."""
    rng = jax.random.key(config.seed) if init_rng is None else init_rng

    def _actor_state_from_rng(rng):
        actor = config.model.create(rng)
        return nnx.state(actor)

    actor_params_shape = jax.eval_shape(_actor_state_from_rng, rng).to_pure_dict()
    loaded_actor_params = _load_weights_and_validate(config.weight_loader, actor_params_shape)
    if loaded_actor_params is None:
        loaded_actor_params = {}

    def init_fn(rng, batch0, loaded):
        return _build_rl_train_state(config, rng, batch0, loaded)

    train_state_shape = jax.eval_shape(init_fn, rng, batch0, loaded_actor_params)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    train_state = jax.jit(
        init_fn,
        donate_argnums=(2,),
        in_shardings=(replicated_sharding, data_sharding, replicated_sharding),
        out_shardings=state_sharding,
    )(rng, batch0, loaded_actor_params)

    # Inject ImageNet-1K pretrained ResNet-10 weights into the critic encoder backbone.
    # Must happen OUTSIDE JIT because it involves file I/O (download / pickle load).
    linen_params = _download_resnet10_params()
    ac = nnx.merge(train_state.model_def, train_state.params)
    for enc in ac.critic_encoder.encoders:
        _map_linen_to_nnx_backbone(linen_params, enc.backbone)
    n_cost_enc = 0
    if config.use_safety_critic and hasattr(ac, 'cost_critic_encoder') and ac.cost_critic_encoder is not ac.critic_encoder:
        for enc in ac.cost_critic_encoder.encoders:
            _map_linen_to_nnx_backbone(linen_params, enc.backbone)
        n_cost_enc = len(ac.cost_critic_encoder.encoders)
    new_params = nnx.state(ac)

    ac_tgt = nnx.merge(train_state.model_def, train_state.target_params)
    for enc in ac_tgt.critic_encoder.encoders:
        _map_linen_to_nnx_backbone(linen_params, enc.backbone)
    if config.use_safety_critic and hasattr(ac_tgt, 'cost_critic_encoder') and ac_tgt.cost_critic_encoder is not ac_tgt.critic_encoder:
        for enc in ac_tgt.cost_critic_encoder.encoders:
            _map_linen_to_nnx_backbone(linen_params, enc.backbone)
    new_target_params = nnx.state(ac_tgt)

    train_state = train_state.replace(params=new_params, target_params=new_target_params)
    logging.info("ResNet-10 pretrained weights injected into %d reward + %d cost critic image encoder(s).",
                 len(ac.critic_encoder.encoders), n_cost_enc)

    # Inject pretrained VAE tactile encoder weights (if tactile is enabled and checkpoint provided).
    # Mirrors the loading pattern in train_tactile_world_model.py Stage 2:
    # create a full TactileBlockVAE → load full ckpt → extract encoder state.
    vae_ckpt = getattr(config, "vae_tactile_ckpt", "")
    tactile_encoder_type = getattr(config, "critic_tactile_encoder_type", "cnn")
    if getattr(config, "critic_encoder_use_tactile", False) and tactile_encoder_type == "vae" and vae_ckpt:
        latent_dim = getattr(config, "vae_tactile_latent_dim", 64)
        loaded_enc_state = _load_vae_encoder_state(vae_ckpt, latent_dim=latent_dim)

        ac2 = nnx.merge(train_state.model_def, train_state.params)
        nnx.update(ac2.critic_encoder.tactile_enc.encoder, loaded_enc_state)
        if config.use_safety_critic and hasattr(ac2, 'cost_critic_encoder') and ac2.cost_critic_encoder is not ac2.critic_encoder:
            nnx.update(ac2.cost_critic_encoder.tactile_enc.encoder, loaded_enc_state)
        new_params2 = nnx.state(ac2)

        ac2_tgt = nnx.merge(train_state.model_def, train_state.target_params)
        nnx.update(ac2_tgt.critic_encoder.tactile_enc.encoder, loaded_enc_state)
        if config.use_safety_critic and hasattr(ac2_tgt, 'cost_critic_encoder') and ac2_tgt.cost_critic_encoder is not ac2_tgt.critic_encoder:
            nnx.update(ac2_tgt.cost_critic_encoder.tactile_enc.encoder, loaded_enc_state)
        new_target_params2 = nnx.state(ac2_tgt)

        train_state = train_state.replace(params=new_params2, target_params=new_target_params2)
        logging.info("Pretrained VAE tactile encoder weights injected into reward + cost encoders (frozen via stop_gradient).")
    elif getattr(config, "critic_encoder_use_tactile", False) and tactile_encoder_type == "vae":
        logging.warning("VAE tactile critic is enabled but vae_tactile_ckpt is empty — tactile VAE encoder has RANDOM weights!")

    return train_state, state_sharding


def _merge_params_shape_tolerant(current_state: nnx.State, saved_raw: Any) -> None:
    """Merge saved params into current_state only where path exists and shape matches."""
    current_pure = current_state.to_pure_dict()
    current_flat = traverse_util.flatten_dict(current_pure)
    flat_saved = traverse_util.flatten_dict(saved_raw)
    if flat_saved and all(kp[-1] == "value" for kp in flat_saved):
        flat_saved = {kp[:-1]: v for kp, v in flat_saved.items()}
    merged_flat = {}
    for path, c_leaf in current_flat.items():
        if path in flat_saved:
            s_leaf = flat_saved[path]
            if hasattr(c_leaf, "shape") and hasattr(s_leaf, "shape") and np.shape(c_leaf) == np.shape(s_leaf):
                merged_flat[path] = s_leaf
                continue
        merged_flat[path] = c_leaf
    current_state.replace_by_pure_dict(traverse_util.unflatten_dict(merged_flat))


def _restore_rl_checkpoint_raw(
    checkpoint_manager: ocp.CheckpointManager,
    step: int,
) -> tuple[Any, Any]:
    """Restore train_state and params using checkpoint structure for shape-tolerant merge."""
    step_dir = epath.Path(checkpoint_manager.directory) / str(step)
    with ocp.PyTreeCheckpointer() as ckptr:
        params_path = step_dir / "params"
        train_state_path = step_dir / "train_state"
        try:
            meta_params = ckptr.metadata(params_path)
            meta_ts = ckptr.metadata(train_state_path)
        except (FileNotFoundError, KeyError, OSError):
            meta_all = ckptr.metadata(step_dir)
            restore_args = jax.tree_util.tree_map(
                lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), meta_all
            )
            restored = ckptr.restore(
                step_dir,
                ocp.args.PyTreeRestore(item=meta_all, restore_args=restore_args),
            )
            saved_params = restored.get("params", {})
            if isinstance(saved_params, dict) and "params" in saved_params:
                saved_params = saved_params["params"]
            return restored.get("train_state", restored), saved_params
        restore_args_p = jax.tree_util.tree_map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), meta_params)
        restore_args_ts = jax.tree_util.tree_map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), meta_ts)
        saved_params_container = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(item=meta_params, restore_args=restore_args_p),
        )
        saved_ts = ckptr.restore(
            train_state_path,
            ocp.args.PyTreeRestore(item=meta_ts, restore_args=restore_args_ts),
        )
    saved_params = saved_params_container.get("params", saved_params_container)
    if isinstance(saved_params, dict) and "params" in saved_params:
        saved_params = saved_params["params"]
    return saved_ts, saved_params


def _get_attr_or_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def restore_state_rl(
    checkpoint_manager: ocp.CheckpointManager,
    state: RLTrainState,
    data_loader: Any,
    step: int | None = None,
) -> RLTrainState:
    """Restore RLTrainState; on shape mismatch do shape-tolerant merge."""
    try:
        return _checkpoints.restore_state(checkpoint_manager, state, data_loader)
    except Exception as e:
        logging.warning("Standard restore failed (%s), attempting shape-tolerant restore.", e)
    step = step if step is not None else max(checkpoint_manager.all_steps())
    saved_ts, saved_params = _restore_rl_checkpoint_raw(checkpoint_manager, step)
    _merge_params_shape_tolerant(state.params, saved_params)
    saved_target = _get_attr_or_key(saved_ts, "target_params")
    if saved_target is not None:
        _merge_params_shape_tolerant(state.target_params, saved_target)
    updates = {}
    step_val = _get_attr_or_key(saved_ts, "step")
    if step_val is not None:
        updates["step"] = jnp.asarray(step_val, dtype=jnp.int32)
    for key in ("opt_state_actor", "opt_state_critic", "opt_state_cost_critic", "lag_lambda"):
        val = _get_attr_or_key(saved_ts, key)
        if val is not None:
            updates[key] = val
    if updates:
        state = state.replace(**updates)
    logging.info("Shape-tolerant restore completed (step=%s). Mismatched params left at init.", int(state.step))
    return state


# ================================================================================================================

def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    # 确认 target_q 裁剪配置：若为 None 则会在 loss 内用默认 100.0，避免 target_qs 爆炸
    logging.info(f"target_q_clip = {config.target_q_clip!r} (TD target 将裁到 ±target_q_clip；None 时 loss 内用 100.0)")
    logging.info(f"reward_bias = {config.reward_bias} (每步入口 batch.rewards = rewards + reward_bias，整条训练使用加过 bias 的 reward，与 SERL 一致)")
    logging.info("reward_critic_encoder = ConRFT PreTrainedResNet-10 (frozen ImageNet-1K backbone + trainable SpatialLearnedEmbeddings + bottleneck)")
    if config.use_safety_critic:
        logging.info("cost_critic_encoder  = ConRFT PreTrainedResNet-10 (SEPARATE instance, frozen backbone + trainable head)")
    logging.info(f"critic_encoder_use_tactile = {getattr(config, 'critic_encoder_use_tactile', False)}, "
                 f"critic_tactile_encoder_type = {getattr(config, 'critic_tactile_encoder_type', 'cnn')!r}, "
                 f"critic_tactile_freeze_backbone = {getattr(config, 'critic_tactile_freeze_backbone', False)}, "
                 f"critic_encoder_tactile_dim = {getattr(config, 'critic_encoder_tactile_dim', 128)}, "
                 f"vae_tactile_ckpt = {getattr(config, 'vae_tactile_ckpt', '')!r}, "
                 f"vae_tactile_latent_dim = {getattr(config, 'vae_tactile_latent_dim', 64)}")
    if getattr(config, "critic_encoder_use_tactile", False):
        if getattr(config, "critic_tactile_encoder_type", "cnn") == "vae":
            logging.info("critic_tactile_encoder = PreTrainedVAETactileEncoder (frozen VAE backbone + trainable projection)")
        else:
            logging.info("critic_tactile_encoder = SingleTactileCNN (direct tactile input, trainable unless critic_tactile_freeze_backbone=True)")
    if config.use_safety_critic:
        if getattr(config, "use_lagrangian", False):
            logging.info(
                f"Safety (cost) critic ENABLED with Lagrangian: "
                f"cost_limit={config.cost_limit}, lag_lambda_init={config.lag_lambda_init}, "
                f"lag_lambda_lr={config.lag_lambda_lr}, lag_lambda_max={config.lag_lambda_max}, "
                f"cost_cql_alpha={config.cost_cql_alpha}, cost_critic_ensemble={config.critic_cost_ensemble}"
            )
        else:
            logging.info(
                f"Safety (cost) critic ENABLED (fixed weight): cost_weight={config.cost_weight}, "
                f"cost_cql_alpha={config.cost_cql_alpha}, cost_critic_ensemble={config.critic_cost_ensemble}"
            )
        logging.info(
            "tactile risk cost: source=%s, f_max=%.3f, asym_delta=%.3f, area_min=%.3f, "
            "contact_threshold=%.3f, weights(force/slip/asym/area)=(%.3f, %.3f, %.3f, %.3f)",
            getattr(config, "tactile_cost_source", "tactile"),
            getattr(config, "tactile_cost_force_max", 4.0),
            getattr(config, "tactile_cost_asym_delta", 2.0),
            getattr(config, "tactile_cost_area_min", 4.0),
            getattr(config, "tactile_cost_contact_threshold", 0.5),
            getattr(config, "tactile_cost_force_weight", 1.0),
            getattr(config, "tactile_cost_slip_weight", 3.0),
            getattr(config, "tactile_cost_asym_weight", 0.1),
            getattr(config, "tactile_cost_area_weight", 100.0),
        )
    if config.critic_warmup_steps > 0:
        logging.info(f"critic_warmup_steps = {config.critic_warmup_steps} (前 {config.critic_warmup_steps} 步仅更新 critic，之后正常 ConRFT)")
        if getattr(config, "critic_warmup_cql_alpha", None) is not None:
            logging.info(f"critic_warmup_cql_alpha = {config.critic_warmup_cql_alpha} (warmup 阶段用此 CQL 权重，便于 Q 收敛到合理尺度)")

    data_loader = create_data_loader_offline_rl(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    
    obs0 = batch.obs
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in obs0.images.values()], axis=1))
        for i in range(min(5, len(next(iter(obs0.images.values())))))
    ]

    # --- init RL train state (FSDP sharding). Always full init so resume can merge. ---
    train_state, train_state_sharding = init_rl_train_state(
        config,
        init_rng,
        batch,
        mesh,
        data_sharding=data_sharding,
        replicated_sharding=replicated_sharding,
        resume=resuming,
    )
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = restore_state_rl(checkpoint_manager, train_state, data_loader)

    wandb.log({"camera_views": images_to_log}, step=int(train_state.step))

    # Resumed state buffers may not be donatable; omit donate_argnums to avoid JAX errors after restore.
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(),
    )
    ptrain_step_critic_only = jax.jit(
        functools.partial(train_step_critic_only, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding, replicated_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(),
    )

    start_step = int(train_state.step)
    num_steps = max(0, config.num_train_steps - start_step)
    if start_step >= config.num_train_steps:
        logging.warning(
            "Resumed at step %s but num_train_steps=%s. No training steps will run. "
            "To continue training, set num_train_steps > %s (e.g. num_train_steps=%s or more).",
            start_step,
            config.num_train_steps,
            start_step,
            start_step + 5000,
        )
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )
    logging.info(
        "Training from step %s to %s (%s steps).",
        start_step,
        config.num_train_steps - 1,
        num_steps,
    )

    # 仅 critic 的 step 不返回 actor 相关 key，stack_forest 要求所有 info 键一致，此处补全缺失键
    INFO_ACTOR_ONLY_KEYS = ("actor_loss", "bc_loss", "q_loss", "q_sampled_mean", "cost_q_term",
                            "effective_cost_weight", "dual_cost_q", "dual_violation")

    CRITIC_PARAM_DIAG_KEYS = frozenset({
        "rc_grad_norm", "rc_update_norm", "rc_param_norm_before", "rc_param_norm_after",
        "rc_param_delta_norm", "rc_delta_critic_mlp", "rc_delta_critic_enc", "rc_n_param_leaves",
        "cc_grad_norm", "cc_update_norm", "cc_param_norm_before", "cc_param_norm_after",
        "cc_param_delta_norm", "cc_delta_cost_mlp", "cc_delta_cost_enc", "cc_n_param_leaves",
    })

    def _normalize_info(info):
        return {**info, **{k: jnp.nan for k in INFO_ACTOR_ONLY_KEYS if k not in info}}

    infos = []
    cta_ratio = max(1, int(config.cta_ratio))
    critic_warmup_steps = int(config.critic_warmup_steps)
    critic_warmup_cql_alpha = getattr(config, "critic_warmup_cql_alpha", None)
    for step in pbar:
        with sharding.set_mesh(mesh):
            if step < start_step + critic_warmup_steps:
                # 先只训 critic：不更新 actor；warmup 阶段可用更小 cql_alpha 让终端 target 把 Q 拉上去
                cql_alpha_use = (
                    critic_warmup_cql_alpha if critic_warmup_cql_alpha is not None else config.cql_alpha
                )
                train_rng, step_rng = jax.random.split(train_rng)
                train_state, info = ptrain_step_critic_only(
                    step_rng, train_state, batch, jnp.array(cql_alpha_use, dtype=jnp.float32)
                )
            else:
                # 正常 ConRFT：(cta_ratio - 1) critic-only updates, then 1 critic+actor
                for _ in range(cta_ratio - 1):
                    train_rng, crng = jax.random.split(train_rng)
                    train_state, _ = ptrain_step_critic_only(
                        crng, train_state, batch, jnp.array(config.cql_alpha, dtype=jnp.float32)
                    )
                    batch = next(data_iter)

                train_rng, step_rng = jax.random.split(train_rng)
                train_state, info = ptrain_step(step_rng, train_state, batch)

        # step = 循环步数（每轮 1 次 actor + (cta_ratio-1) 次 critic），与 save_interval/num_train_steps 一致
        train_state = train_state.replace(step=jnp.int32(step + 1))

        infos.append(_normalize_info(info))

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.nanmean, stacked_infos))
            compact = {k: v for k, v in reduced_info.items() if k not in CRITIC_PARAM_DIAG_KEYS}
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(compact.items()))
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            if getattr(config, "log_critic_param_diagnostics", True):
                r = reduced_info
                logging.info(
                    "[critic param update] reward critic: grad_L2=%.6f update_L2=%.6f param_delta_L2=%.6f "
                    "(mlp_delta=%.6f encoder_delta=%.6f n_leaves=%.0f)",
                    float(r.get("rc_grad_norm", float("nan"))),
                    float(r.get("rc_update_norm", float("nan"))),
                    float(r.get("rc_param_delta_norm", float("nan"))),
                    float(r.get("rc_delta_critic_mlp", float("nan"))),
                    float(r.get("rc_delta_critic_enc", float("nan"))),
                    float(r.get("rc_n_param_leaves", float("nan"))),
                )
                logging.info(
                    "[critic param update] cost critic:   grad_L2=%.6f update_L2=%.6f param_delta_L2=%.6f "
                    "(mlp_delta=%.6f encoder_delta=%.6f n_leaves=%.0f)",
                    float(r.get("cc_grad_norm", float("nan"))),
                    float(r.get("cc_update_norm", float("nan"))),
                    float(r.get("cc_param_delta_norm", float("nan"))),
                    float(r.get("cc_delta_cost_mlp", float("nan"))),
                    float(r.get("cc_delta_cost_enc", float("nan"))),
                    float(r.get("cc_n_param_leaves", float("nan"))),
                )
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
