VTLA (Vision-Tactile-Language-Action) Guide
===========================================

This document describes the VTLA implementation built on top of openpi PI0.5 with
an additional tactile encoder. It covers the design, fine-tuning from the PI0.5
base model, the tactile LeRobot dataset format, and preprocessing details. The
content follows the current implementation in this repository.

1. VTLA Design Overview
-----------------------

VTLA treats tactile observations as additional conditioning tokens. These tactile
tokens are concatenated with visual and language tokens as prefix conditions for
the action expert:

- Visual tokens: SigLIP patch tokens in the prefix.
- Language tokens: PaliGemma tokenizer outputs in the prefix.
- Tactile tokens: 4 tokens for each hand, 8 tactile tokens in total, in the prefix.
- Action tokens: continuous actions predicted by the action expert in the suffix.

The training objective remains the PI0.5 flow-matching MSE for action regression.
The tactile encoder only provides conditional information and is updated through
the same loss. When only the tactile encoder is trainable and all other modules
are frozen, it can still learn useful tactile representations through the action
prediction error.

Key implementation points:

- Tactile encoder: `src/openpi/models/tactile_encoder.py`
- Token concatenation: `embed_prefix` in `src/openpi/models/pi0.py`
- Offline TacRL / Cal-QL: `scripts/train_tacrl_offline.py`
- Consistent data pipeline: training and inference both use `data_transforms.inputs`
- Preprocessing and normalization: baseline subtraction + log1p + Normalize


2. Fine-Tuning from PI0.5 Base Model
------------------------------------

This section describes the PI0.5 fine-tuning flow from the base model using the
repository configuration `uf850_pi05_lora_tactile`. In this setup, only the
tactile encoder is trained and all other modules are frozen.

2.1 Environment Setup
^^^^^^^^^^^^^^^^^^^^^

Make sure the JAX/Flax runtime is available. Training cannot start if the current
environment does not have `jax` / `jaxlib` installed.

2.2 Compute Norm Stats, Including Tactile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tactile data must be included in the norm statistics. Otherwise, `Normalize` will
not be applied to tactile inputs.

Run:

```
python scripts/compute_norm_stats.py --config-name uf850_pi05_lora_tactile
```

The script computes statistics for the following keys:

`state`, `actions`, `tactile_left`, `tactile_right`

2.3 Start Fine-Tuning
^^^^^^^^^^^^^^^^^^^^^

```
python scripts/train.py --config-name uf850_pi05_lora_tactile --exp-name tactile_ft
```

Notes:

- The default checkpoint is `gs://openpi-assets/checkpoints/pi05_base/params`.
- Only the tactile encoder is trained: `freeze_filter=Not(PathRegex(".*tactile.*"))`.
- EMA is disabled with `ema_decay=None`.

2.4 Training and Inference Consistency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Training and inference use the same transform pipeline:

1. `TactilePreprocess`: baseline subtraction + log1p.
2. `UF850Inputs`: converts dataset keys into the model input structure.
3. `Normalize`: uses the computed norm statistics.

During inference, `Policy` calls `data_transforms.inputs`, which keeps the
preprocessing path consistent with training.


3. LeRobot Dataset Format with Tactile Inputs
---------------------------------------------

3.1 Required Fields
^^^^^^^^^^^^^^^^^^^

Each sample, using flat keys, should contain at least:

- `observation.images.front`       (uint8, HxWx3)
- `observation.images.wrist`       (uint8, HxWx3)
- `observation.state`              (float32, action_dim)
- `observation.tactile_left`       (float32, [T, 16, 16])
- `observation.tactile_right`      (float32, [T, 16, 16])
- `action`                         (float32, action_dim)
- `task_index`                     (int, required when `prompt_from_task=True`)

Notes:

- The default value of T is 5. If T is changed to 8, update both the dataset and
  the configuration.
- Tactile keys use `observation/tactile_left/right` during repacking.

3.2 Example `meta/modality.json`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

```
{
  "observation.images.front": {"type": "image", "shape": [224, 224, 3], "dtype": "uint8"},
  "observation.images.wrist": {"type": "image", "shape": [224, 224, 3], "dtype": "uint8"},
  "observation.state": {"type": "float32", "shape": [7]},
  "observation.tactile_left": {"type": "float32", "shape": [5, 16, 16]},
  "observation.tactile_right": {"type": "float32", "shape": [5, 16, 16]},
  "action": {"type": "float32", "shape": [7]},
  "task_index": {"type": "int32", "shape": []}
}
```

3.3 Example `meta/info.json`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

```
{
  "fps": 30,
  "action_dim": 7,
  "total_episodes": 34,
  "total_frames": 7411,
  "tasks": {
    "0": "pick up the plate"
  }
}
```

3.4 Tactile Preprocessing Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current tactile preprocessing implementation uses:

1. Baseline subtraction: the first tactile frame in each sample is used as the baseline.
2. log1p: values are clipped to `>= 0` before `log1p` to avoid NaNs from negative values.
3. Normalize: mean and variance are computed by `compute_norm_stats.py`.

If the tactile data contains meaningful negative values and the sign should be
preserved, this step can be changed to `signed_log1p`.


4. Tactile Model Structure and Training Logic
---------------------------------------------

- The tactile encoder outputs 4 tokens for each hand, 8 tokens in total, and
  `embed_prefix` appends them to the end of the prefix.
- The loss remains the PI0.5 flow-matching MSE. No additional loss is introduced.
- When the rest of the model is frozen, gradients only update the tactile encoder
  parameters.


5. Paper-Style Offline TacRL Training
-------------------------------------

Offline RL starts from `scripts/train_tacrl_offline.py`. The script implements the
paper-style reward critic `Q_R`, safety/cost critic `Q_C`, Cal-QL critic loss, and
the actor-side Lagrangian constrained objective:

```
L_actor = beta * L_BC - eta * Q_R + lambda * Q_C
```

Recommended entry point:

```
bash sh/rl.sh
```

Or run the script directly:

```
uv run scripts/train_tacrl_offline.py uf850_pi05_lora_tactile_rl \
  --exp-name=tacrl_offline \
  --use_safety_critic=True \
  --use_lagrangian=True \
  --cost_limit=4 \
  --tactile_cost_source=tactile
```

5.1 Critics Use Tactile Inputs by Default
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The TacRL critics use tactile inputs by default through
`critic_encoder_use_tactile=True` and `critic_tactile_encoder_type="cnn"`.
This means both the reward critic and the cost critic read raw
`observation.tactile_left/right` values directly. Each critic encodes tactile
inputs with its own `SingleTactileCNN`, then concatenates the result into the
critic state. This CNN is trainable by default
(`critic_tactile_freeze_backbone=False`) and no longer depends on a VAE checkpoint
by default.

A VAE checkpoint is only needed for VAE ablations. In that case, set:

```
--critic_tactile_encoder_type=vae --vae_tactile_ckpt=...
```

5.2 Four Tactile Safety Cost Terms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When `use_safety_critic=True`, the training script no longer depends on a
pre-labeled `cost` field in the dataset by default. Instead, it computes a
continuous tactile risk signal online from `observation.tactile_left/right`
following the paper formula:

```
C = w_force * ReLU(f - f_max)^2
  + w_slip  * ||CoP_t - CoP_{t-1}||^2
  + w_asym  * ReLU(|f_L - f_R| - delta)^2
  + w_area  * ReLU(A_min - A)^2
```

Default weights follow the paper appendix:

```
tactile_cost_force_weight = 1.0
tactile_cost_slip_weight  = 3.0
tactile_cost_asym_weight  = 0.1
tactile_cost_area_weight  = 100.0
```

The corresponding thresholds can be adjusted in `TrainConfig` or from the
command line:

`tactile_cost_force_max`, `tactile_cost_asym_delta`,
`tactile_cost_area_min`, `tactile_cost_contact_threshold`

For the old-style ablation, set `--tactile_cost_source=dataset`. In that mode,
the script reads the dataset `cost` field and binarizes it using
`cost_binarize_threshold`.

5.3 Online Fine-Tuning with Two Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The online-stage entry point is `scripts/train_rft_online.py`. It mixes the
original expert/demo dataset with a second online-collected dataset. Each batch is
sampled according to `dataset_a_ratio`. Critic updates use pure TD loss and
explicitly disable Cal-QL / CQL regularization.

```
uv run scripts/train_rft_online.py uf850_pi05_lora_tactile_rl \
  --exp-name=tacrl_online \
  --second-data-repo=/path/to/online_lerobot_dataset \
  --dataset-a-ratio=0.5
```


6. Common Adjustments
---------------------

6.1 Change T from 5 to 8
^^^^^^^^^^^^^^^^^^^^^^^^

Update:

- `Pi0Config.tactile_T`
- Dataset `modality.json` and the actual tactile data shape
- Any scripts or assertions that hard-code T

No changes are needed for:

- `tactile_encoder`, which already supports arbitrary prefix dimensions
- Token concatenation logic

6.2 Multi-GPU Batch Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`tactile_encoder` supports inputs with shape `(devices, B, T, H, W)`. It preserves
all batch prefix dimensions through:

```
batch_shape = x.shape[:-3]
```


7. Key Files
------------

- Tactile encoder: `src/openpi/models/tactile_encoder.py`
- Token concatenation: `src/openpi/models/pi0.py`
- Configuration and freezing: `src/openpi/training/config.py`
- Offline TacRL: `scripts/train_tacrl_offline.py`
- Online two-dataset fine-tuning: `scripts/train_rft_online.py`
- Tactile preprocessing: `src/openpi/transforms.py`
- Norm statistics script: `scripts/compute_norm_stats.py`
