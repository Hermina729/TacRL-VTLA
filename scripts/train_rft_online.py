"""Online-stage TacRL fine-tuning from two pre-collected datasets.

This script is the online-stage training entrypoint described in the thesis:
it mixes the original demonstration dataset with a second dataset collected
after deployment / intervention, and updates the same reward-cost actor-critic
used by offline TacRL.

Unlike offline TacRL, online critic updates use pure TD losses: Cal-QL / CQL
regularization is disabled here by forcing ``disable_calql=True`` and setting
both reward and cost CQL weights to zero.

Example:
    uv run scripts/train_rft_online.py uf850_pi05_lora_tactile_rl \
        --exp-name=online_dual_dataset \
        --second-data-repo=/path/to/online_lerobot_dataset \
        --dataset-a-ratio=0.5
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import logging
import platform
import sys
from typing import Any

import etils.epath as epath
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

from train_tacrl_offline import (
    RLBatch,
    create_data_loader_offline_rl,
    init_logging,
    init_rl_train_state,
    init_wandb,
    restore_state_rl,
    train_step,
    train_step_critic_only,
)


def _as_jax_batch_slice(x: Any, start: int, end: int):
    return jnp.asarray(x)[start:end]


def _slice_obs(obs: _model.Observation, start: int, end: int) -> _model.Observation:
    images = {k: _as_jax_batch_slice(v, start, end) for k, v in obs.images.items()}
    image_masks = {k: _as_jax_batch_slice(v, start, end) for k, v in obs.image_masks.items()}
    kw: dict[str, Any] = {
        "images": images,
        "image_masks": image_masks,
        "state": _as_jax_batch_slice(obs.state, start, end),
    }
    for name in (
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "token_ar_mask",
        "token_loss_mask",
        "tactile_left",
        "tactile_right",
    ):
        value = getattr(obs, name)
        kw[name] = _as_jax_batch_slice(value, start, end) if value is not None else None
    return obs.replace(**kw)


def slice_rl_batch(batch: RLBatch, start: int, end: int) -> RLBatch:
    """Slice an RL batch on axis 0."""
    return batch.replace(
        obs=_slice_obs(batch.obs, start, end),
        next_obs=_slice_obs(batch.next_obs, start, end),
        actions=_as_jax_batch_slice(batch.actions, start, end),
        rewards=_as_jax_batch_slice(batch.rewards, start, end),
        dones=_as_jax_batch_slice(batch.dones, start, end),
        masks=_as_jax_batch_slice(batch.masks, start, end),
        tasks=_as_jax_batch_slice(batch.tasks, start, end) if batch.tasks is not None else None,
        intervened=_as_jax_batch_slice(batch.intervened, start, end) if batch.intervened is not None else None,
        mc_returns=_as_jax_batch_slice(batch.mc_returns, start, end) if batch.mc_returns is not None else None,
        embeddings=_as_jax_batch_slice(batch.embeddings, start, end) if batch.embeddings is not None else None,
        next_embeddings=(
            _as_jax_batch_slice(batch.next_embeddings, start, end)
            if batch.next_embeddings is not None
            else None
        ),
        costs=_as_jax_batch_slice(batch.costs, start, end) if batch.costs is not None else None,
        cost_mc_returns=(
            _as_jax_batch_slice(batch.cost_mc_returns, start, end)
            if batch.cost_mc_returns is not None
            else None
        ),
    )


def _concat_optional(a: Any, b: Any, *, name: str) -> Any:
    if a is None and b is None:
        return None
    if a is None or b is None:
        raise ValueError(f"{name} exists on only one dataset; both dataset transforms must match.")
    return jnp.concatenate([jnp.asarray(a), jnp.asarray(b)], axis=0)


def _concat_obs(a: _model.Observation, b: _model.Observation) -> _model.Observation:
    if set(a.images) != set(b.images):
        raise ValueError(f"Image keys differ: A={sorted(a.images)} B={sorted(b.images)}")
    if set(a.image_masks) != set(b.image_masks):
        raise ValueError(f"Image mask keys differ: A={sorted(a.image_masks)} B={sorted(b.image_masks)}")
    return a.replace(
        images={k: jnp.concatenate([jnp.asarray(a.images[k]), jnp.asarray(b.images[k])], axis=0) for k in a.images},
        image_masks={
            k: jnp.concatenate([jnp.asarray(a.image_masks[k]), jnp.asarray(b.image_masks[k])], axis=0)
            for k in a.image_masks
        },
        state=jnp.concatenate([jnp.asarray(a.state), jnp.asarray(b.state)], axis=0),
        tokenized_prompt=_concat_optional(a.tokenized_prompt, b.tokenized_prompt, name="tokenized_prompt"),
        tokenized_prompt_mask=_concat_optional(
            a.tokenized_prompt_mask, b.tokenized_prompt_mask, name="tokenized_prompt_mask"
        ),
        token_ar_mask=_concat_optional(a.token_ar_mask, b.token_ar_mask, name="token_ar_mask"),
        token_loss_mask=_concat_optional(a.token_loss_mask, b.token_loss_mask, name="token_loss_mask"),
        tactile_left=_concat_optional(a.tactile_left, b.tactile_left, name="tactile_left"),
        tactile_right=_concat_optional(a.tactile_right, b.tactile_right, name="tactile_right"),
    )


def concat_rl_batches(a: RLBatch, b: RLBatch) -> RLBatch:
    return RLBatch(
        obs=_concat_obs(a.obs, b.obs),
        actions=jnp.concatenate([jnp.asarray(a.actions), jnp.asarray(b.actions)], axis=0),
        rewards=jnp.concatenate([jnp.asarray(a.rewards), jnp.asarray(b.rewards)], axis=0),
        dones=jnp.concatenate([jnp.asarray(a.dones), jnp.asarray(b.dones)], axis=0),
        masks=jnp.concatenate([jnp.asarray(a.masks), jnp.asarray(b.masks)], axis=0),
        next_obs=_concat_obs(a.next_obs, b.next_obs),
        tasks=_concat_optional(a.tasks, b.tasks, name="tasks"),
        intervened=_concat_optional(a.intervened, b.intervened, name="intervened"),
        mc_returns=_concat_optional(a.mc_returns, b.mc_returns, name="mc_returns"),
        embeddings=_concat_optional(a.embeddings, b.embeddings, name="embeddings"),
        next_embeddings=_concat_optional(a.next_embeddings, b.next_embeddings, name="next_embeddings"),
        costs=_concat_optional(a.costs, b.costs, name="costs"),
        cost_mc_returns=_concat_optional(a.cost_mc_returns, b.cost_mc_returns, name="cost_mc_returns"),
    )


class DualDatasetLoader:
    """Mix two RL dataloaders at a fixed ratio in each local batch."""

    def __init__(self, loader_a, loader_b, *, dataset_a_ratio: float):
        self._loader_a = loader_a
        self._loader_b = loader_b
        self._dataset_a_ratio = float(dataset_a_ratio)
        self._iter_a = None
        self._iter_b = None

    def data_config(self):
        return self._loader_a.data_config()

    def __iter__(self):
        self._iter_a = iter(self._loader_a)
        self._iter_b = iter(self._loader_b)
        return self

    def __next__(self) -> RLBatch:
        try:
            batch_a = next(self._iter_a)
        except StopIteration:
            self._iter_a = iter(self._loader_a)
            batch_a = next(self._iter_a)
        try:
            batch_b = next(self._iter_b)
        except StopIteration:
            self._iter_b = iter(self._loader_b)
            batch_b = next(self._iter_b)

        batch_size = int(batch_a.actions.shape[0])
        if int(batch_b.actions.shape[0]) != batch_size:
            raise ValueError(
                f"Local batch sizes differ: A={batch_size}, B={int(batch_b.actions.shape[0])}. "
                "Use matching config.batch_size and device count for both datasets."
            )

        ratio = max(0.0, min(1.0, self._dataset_a_ratio))
        n_a = max(0, min(batch_size, int(round(batch_size * ratio))))
        n_b = batch_size - n_a
        if n_a == 0:
            return slice_rl_batch(batch_b, 0, batch_size)
        if n_b == 0:
            return slice_rl_batch(batch_a, 0, batch_size)
        return concat_rl_batches(slice_rl_batch(batch_a, 0, n_a), slice_rl_batch(batch_b, 0, n_b))


def _put(x: Any, data_sharding: jax.sharding.Sharding):
    return jax.device_put(jnp.asarray(x), data_sharding)


def _shard_obs(obs: _model.Observation, data_sharding: jax.sharding.Sharding) -> _model.Observation:
    kw = {
        "images": {k: _put(v, data_sharding) for k, v in obs.images.items()},
        "image_masks": {k: _put(v, data_sharding) for k, v in obs.image_masks.items()},
        "state": _put(obs.state, data_sharding),
    }
    for name in (
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "token_ar_mask",
        "token_loss_mask",
        "tactile_left",
        "tactile_right",
    ):
        value = getattr(obs, name)
        kw[name] = _put(value, data_sharding) if value is not None else None
    return obs.replace(**kw)


def shard_rl_batch(batch: RLBatch, data_sharding: jax.sharding.Sharding) -> RLBatch:
    return batch.replace(
        obs=_shard_obs(batch.obs, data_sharding),
        next_obs=_shard_obs(batch.next_obs, data_sharding),
        actions=_put(batch.actions, data_sharding),
        rewards=_put(batch.rewards, data_sharding),
        dones=_put(batch.dones, data_sharding),
        masks=_put(batch.masks, data_sharding),
        tasks=_put(batch.tasks, data_sharding) if batch.tasks is not None else None,
        intervened=_put(batch.intervened, data_sharding) if batch.intervened is not None else None,
        mc_returns=_put(batch.mc_returns, data_sharding) if batch.mc_returns is not None else None,
        embeddings=_put(batch.embeddings, data_sharding) if batch.embeddings is not None else None,
        next_embeddings=_put(batch.next_embeddings, data_sharding) if batch.next_embeddings is not None else None,
        costs=_put(batch.costs, data_sharding) if batch.costs is not None else None,
        cost_mc_returns=_put(batch.cost_mc_returns, data_sharding) if batch.cost_mc_returns is not None else None,
    )


def next_rl_batch_sharded(data_iter: Any, data_sharding: jax.sharding.Sharding) -> RLBatch:
    return shard_rl_batch(next(data_iter), data_sharding)


def make_online_config(config: _config.TrainConfig) -> _config.TrainConfig:
    """Apply online-stage overrides from the thesis: no Cal-QL regularization."""
    return dataclasses.replace(
        config,
        disable_calql=True,
        cql_alpha=0.0,
        cost_cql_alpha=0.0,
        critic_warmup_cql_alpha=0.0,
    )


def main(
    config: _config.TrainConfig,
    *,
    second_data_repo: str | None = None,
    dataset_a_ratio: float = 0.5,
):
    init_logging()
    config = make_online_config(config)
    logging.info("Running on: %s", platform.node())
    logging.info(
        "Online TacRL dual-dataset mode: second_data_repo=%r, dataset_a_ratio=%.3f, "
        "disable_calql=%s, cql_alpha=%.3f, cost_cql_alpha=%.3f",
        second_data_repo,
        dataset_a_ratio,
        config.disable_calql,
        config.cql_alpha,
        config.cost_cql_alpha,
    )

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

    loader_a = create_data_loader_offline_rl(config, sharding=data_sharding, shuffle=True)
    if second_data_repo:
        data_b = dataclasses.replace(config.data, repo_id=second_data_repo)
        config_b = dataclasses.replace(config, data=data_b)
        loader_b = create_data_loader_offline_rl(config_b, sharding=data_sharding, shuffle=True)
        data_loader = DualDatasetLoader(loader_a, loader_b, dataset_a_ratio=dataset_a_ratio)
    else:
        logging.warning("No --second-data-repo was provided; training will use only the config dataset.")
        data_loader = loader_a

    data_iter = iter(data_loader)
    batch = next_rl_batch_sharded(data_iter, data_sharding)
    logging.info("Initialized data loader:\n%s", training_utils.array_tree_to_info(batch))

    obs0 = batch.obs
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in obs0.images.values()], axis=1))
        for i in range(min(5, len(next(iter(obs0.images.values())))))
    ]

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
    logging.info("Initialized train state:\n%s", training_utils.array_tree_to_info(train_state.params))

    if resuming:
        train_state = restore_state_rl(checkpoint_manager, train_state, data_loader)

    wandb.log({"camera_views": images_to_log}, step=int(train_state.step))

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
    if start_step >= config.num_train_steps:
        logging.warning(
            "Resumed at step %s but num_train_steps=%s. Nothing to train.",
            start_step,
            config.num_train_steps,
        )

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )
    logging.info("Training from step %s to %s.", start_step, config.num_train_steps - 1)

    info_actor_only_keys = (
        "actor_loss",
        "bc_loss",
        "q_loss",
        "q_sampled_mean",
        "cost_q_term",
        "effective_cost_weight",
        "dual_cost_q",
        "dual_violation",
    )

    def _normalize_info(info):
        return {**info, **{k: jnp.nan for k in info_actor_only_keys if k not in info}}

    infos = []
    cta_ratio = max(1, int(config.cta_ratio))
    critic_warmup_steps = int(config.critic_warmup_steps)
    zero_cql_alpha = jnp.array(0.0, dtype=jnp.float32)

    for step in pbar:
        with sharding.set_mesh(mesh):
            if step < start_step + critic_warmup_steps:
                train_rng, step_rng = jax.random.split(train_rng)
                train_state, info = ptrain_step_critic_only(step_rng, train_state, batch, zero_cql_alpha)
            else:
                for _ in range(cta_ratio - 1):
                    train_rng, crng = jax.random.split(train_rng)
                    train_state, _ = ptrain_step_critic_only(crng, train_state, batch, zero_cql_alpha)
                    batch = next_rl_batch_sharded(data_iter, data_sharding)

                train_rng, step_rng = jax.random.split(train_rng)
                train_state, info = ptrain_step(step_rng, train_state, batch)

        train_state = train_state.replace(step=jnp.int32(step + 1))
        infos.append(_normalize_info(info))

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.nanmean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(reduced_info.items()))
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        batch = next_rl_batch_sharded(data_iter, data_sharding)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--second-data-repo", "--online-data-dir", "--online_data_dir", type=str, default=None)
    parser.add_argument("--dataset-a-ratio", "--demo-ratio", "--demo_ratio", type=float, default=0.5)
    known, rest = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]
    main(
        _config.cli(),
        second_data_repo=known.second_data_repo,
        dataset_a_ratio=float(known.dataset_a_ratio),
    )
