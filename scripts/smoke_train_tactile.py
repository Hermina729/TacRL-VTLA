import argparse
import logging

import jax
import jax.numpy as jnp
from flax import nnx

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.weight_loaders as weight_loaders

# Reuse train utilities from the main training script.
from scripts import train as train_script


def _make_fake_batch(model_config: pi0_config.Pi0Config, batch_size: int):
    dataset = _data_loader.FakeDataset(model_config, num_samples=batch_size)
    samples = [dataset[i] for i in range(batch_size)]
    batch = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *samples)
    return _model.Observation.from_dict(batch), batch["actions"]


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2, help="Number of training steps to run.")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    model_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_dim=7,
        action_horizon=2,
        discrete_state_input=False,
        use_tactile=True,
        tactile_T=5,
        tactile_H=16,
        tactile_W=16,
        tactile_use_delta=True,
        tactile_hidden=1024,
        tactile_emb_dim=512,
        tactile_dropout=0.10,
        # Use dummy variants for a lightweight smoke test.
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )

    train_cfg = _config.TrainConfig(
        name="smoke_tactile",
        exp_name="smoke_tactile",
        model=model_cfg,
        data=_config.FakeDataConfig(),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        ema_decay=None,
        batch_size=args.batch_size,
        num_train_steps=args.steps,
        # Freeze everything except tactile parameters.
        freeze_filter=nnx.Not(nnx_utils.PathRegex(".*tactile.*")),
    )

    rng = jax.random.key(0)
    mesh = sharding.make_mesh(num_fsdp_devices=1)

    with sharding.set_mesh(mesh):
        state, _ = train_script.init_train_state(train_cfg, rng, mesh, resume=False)

    model = nnx.merge(state.model_def, state.params)
    trainable_state = nnx.state(model, train_cfg.trainable_filter).flat_state()
    non_tactile = ["/".join(map(str, k)) for k in trainable_state if "tactile" not in "/".join(map(str, k))]
    if non_tactile:
        raise RuntimeError(f"Non-tactile params are trainable: {non_tactile[:5]}")

    for step in range(args.steps):
        rng, step_rng = jax.random.split(rng)
        batch = _make_fake_batch(model_cfg, args.batch_size)
        state, info = train_script.train_step(train_cfg, step_rng, state, batch)
        logging.info("step=%d loss=%.4f grad_norm=%.4f", step, float(info["loss"]), float(info["grad_norm"]))

    logging.info("Smoke test finished. Tactile-only training path is runnable.")


if __name__ == "__main__":
    main()
