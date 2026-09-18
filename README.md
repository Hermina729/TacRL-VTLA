VTLA (Vision-Tactile-Language-Action) 说明与使用
=================================================

本文档聚焦在 openpi 的 PI0.5 基础上加入触觉 encoder 的 VTLA 方案，包括设计原理、从 base 模型进行
PI0.5 微调的流程，以及带 tactile 的 LeRobot 数据集格式与预处理细节。内容以当前仓库实现为准。

1. VTLA 设计原理概述
--------------------

VTLA 的核心思想是将触觉作为额外的条件 token，和视觉 + 语言一起作为 prefix 条件输入动作专家：

- 视觉 token：来自 SigLIP 的 patch token（prefix）
- 语言 token：来自 PaliGemma tokenizer（prefix）
- 触觉 token：左右手各 4 个 token，共 8 个 tactile tokens（prefix）
- 动作 token：action expert 输出的连续动作（suffix）

训练目标仍是 PI0.5 的 flow matching MSE（动作回归），触觉 encoder 只提供条件信息并通过同一 loss
反向传播更新。若只训练触觉 encoder，其他模块冻结，仍可通过动作误差学习有用的触觉表征。

关键实现点：
- 触觉 encoder：`src/openpi/models/tactile_encoder.py`
- Token 拼接：`src/openpi/models/pi0.py` 的 `embed_prefix`
- 离线 TacRL / Cal-QL：`scripts/train_tacrl_offline.py`
- 数据管线一致：训练与推理均使用同一 `data_transforms.inputs`
- 预处理与归一化：baseline subtraction + log1p + Normalize


2. 从 PI0.5 Base 进行微调（仅触觉可训练）
------------------------------------------

本节只描述基于 base 模型的 PI0.5 微调流程，使用仓库已定义配置：
`uf850_pi05_lora_tactile`（仅训练触觉 encoder，其余全部冻结）。

2.1 环境准备
^^^^^^^^^^^^

请确保 JAX/Flax 运行环境可用（当前环境未安装 jax/jaxlib 时无法启动训练）。

2.2 计算 norm stats（包含 tactile）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

触觉数据需要统计到 norm stats，否则 Normalize 不会对 tactile 生效。

执行：
```
python scripts/compute_norm_stats.py --config-name uf850_pi05_lora_tactile
```

该脚本会统计以下 keys：
`state`, `actions`, `tactile_left`, `tactile_right`

2.3 启动微调
^^^^^^^^^^^^

```
python scripts/train.py --config-name uf850_pi05_lora_tactile --exp-name tactile_ft
```

说明：
- 默认加载 `gs://openpi-assets/checkpoints/pi05_base/params`
- 仅训练触觉 encoder：配置中 `freeze_filter=Not(PathRegex(".*tactile.*"))`
- EMA 关闭（`ema_decay=None`）

2.4 训练/推理一致性
^^^^^^^^^^^^^^^^^^^

训练与推理使用同一条 transforms 管线：
1) `TactilePreprocess`（baseline subtraction + log1p）
2) `UF850Inputs`（将 dataset keys 转为模型输入结构）
3) `Normalize`（根据计算的 norm stats）

推理路径由 `Policy` 统一调用 `data_transforms.inputs`，确保和训练一致。


3. 加入触觉后 LeRobot 数据集格式
-------------------------------

3.1 必要字段
^^^^^^^^^^^^

每条样本（扁平 key）需要至少包含：
- `observation.images.front`       (uint8, HxWx3)
- `observation.images.wrist`       (uint8, HxWx3)
- `observation.state`              (float32, action_dim)
- `observation.tactile_left`       (float32, [T, 16, 16])
- `observation.tactile_right`      (float32, [T, 16, 16])
- `action`                         (float32, action_dim)
- `task_index`                     (int，prompt_from_task=True 时必须)

备注：
- T 默认是 5；如果改为 8，需要同步更新数据集和配置。
- 触觉 key 使用 `observation/tactile_left/right`（repack 时会映射）。

3.2 `meta/modality.json` 示例
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

3.3 `meta/info.json` 示例
^^^^^^^^^^^^^^^^^^^^^^^^^

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

3.4 预处理细节（触觉）
^^^^^^^^^^^^^^^^^^^^^

当前实现的触觉预处理：
1) baseline subtraction：每条样本使用第 1 帧作为基线
2) log1p：为了避免负数 NaN，先截断到 >= 0
3) Normalize：使用 `compute_norm_stats.py` 统计出的均值/方差

如果你的触觉数据存在显著负值并希望保留符号，可改为 `signed_log1p`。


4. 触觉模型结构与训练逻辑说明
----------------------------

- 触觉 encoder 输出左右手各 4 个 token，共 8 个 token，通过 `embed_prefix` 拼到 prefix 末尾
- Loss 仍是 PI0.5 flow-matching MSE，不新增额外 loss
- 当冻结其余模块时，梯度仅更新 tactile encoder 参数


5. 论文版离线 TacRL 训练
-----------------------

离线 RL 入口使用 `scripts/train_tacrl_offline.py`。该脚本实现论文中的 reward critic
`Q_R`、safety/cost critic `Q_C`、Cal-QL critic loss，以及 actor 侧的 Lagrangian 约束目标：

```
L_actor = beta * L_BC - eta * Q_R + lambda * Q_C
```

推荐入口：

```
bash sh/rl.sh
```

或直接运行：

```
uv run scripts/train_tacrl_offline.py uf850_pi05_lora_tactile_rl \
  --exp-name=tacrl_offline \
  --use_safety_critic=True \
  --use_lagrangian=True \
  --cost_limit=4 \
  --tactile_cost_source=tactile
```

5.1 Critic 默认直接使用 tactile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TacRL critic 默认开启 `critic_encoder_use_tactile=True`，并使用
`critic_tactile_encoder_type="cnn"`。也就是说 reward critic 和 cost critic 都会直接从
`observation.tactile_left/right` 读取 raw tactile，经各自的 `SingleTactileCNN` 编码后拼进
critic state；该 CNN 默认可训练（`critic_tactile_freeze_backbone=False`），不再默认依赖
VAE checkpoint。只有做 VAE ablation 时才需要显式设置
`--critic_tactile_encoder_type=vae --vae_tactile_ckpt=...`。

5.2 四项 tactile safety cost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`use_safety_critic=True` 时，默认不再依赖数据集中预先标注的 `cost` 字段，而是从
`observation.tactile_left/right` 按论文公式实时计算连续风险：

```
C = w_force * ReLU(f - f_max)^2
  + w_slip  * ||CoP_t - CoP_{t-1}||^2
  + w_asym  * ReLU(|f_L - f_R| - delta)^2
  + w_area  * ReLU(A_min - A)^2
```

默认权重与论文 appendix 对齐：

```
tactile_cost_force_weight = 1.0
tactile_cost_slip_weight  = 3.0
tactile_cost_asym_weight  = 0.1
tactile_cost_area_weight  = 100.0
```

相关阈值可在 `TrainConfig` 或命令行中调整：
`tactile_cost_force_max`、`tactile_cost_asym_delta`、`tactile_cost_area_min`、
`tactile_cost_contact_threshold`。如果要做旧版 ablation，可设置
`--tactile_cost_source=dataset`，此时脚本会读取 dataset 的 `cost` 字段并按
`cost_binarize_threshold` 二值化。

5.3 Online 阶段双数据集微调
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Online 阶段入口是 `scripts/train_rft_online.py`。该脚本混合原 expert/demo 数据集和第二路
online-collected 数据集，每个 batch 按 `dataset_a_ratio` 取样；critic 更新使用纯 TD loss，
显式关闭 Cal-QL / CQL regularization。

```
uv run scripts/train_rft_online.py uf850_pi05_lora_tactile_rl \
  --exp-name=tacrl_online \
  --second-data-repo=/path/to/online_lerobot_dataset \
  --dataset-a-ratio=0.5
```


6. 常见调整
-----------

6.1 T 从 5 改到 8
^^^^^^^^^^^^^^^^

需要改：
- `Pi0Config.tactile_T`
- 数据集 `modality.json` 与实际数据 shape
- 若有硬编码断言/脚本，需同步改动

不需要改：
- `tactile_encoder` 逻辑（已支持任意前缀维）
- token 拼接逻辑

6.2 多卡训练 batch 维度
^^^^^^^^^^^^^^^^^^^^^^^

`tactile_encoder` 已支持 `(devices, B, T, H, W)` 形式，
通过 `batch_shape = x.shape[:-3]` 保留所有 batch 前缀维度。


7. 关键文件索引
---------------

- 触觉 encoder：`src/openpi/models/tactile_encoder.py`
- Token 拼接：`src/openpi/models/pi0.py`
- 配置与冻结：`src/openpi/training/config.py`
- 离线 TacRL：`scripts/train_tacrl_offline.py`
- Online 双数据集微调：`scripts/train_rft_online.py`
- 触觉预处理：`src/openpi/transforms.py`
- 统计脚本：`scripts/compute_norm_stats.py`
