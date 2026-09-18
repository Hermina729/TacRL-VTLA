#!/bin/bash
#SBATCH --job-name=vtla_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00
#SBATCH --output=exp/slurm_logs/%x_%j.out
#SBATCH --error=exp/slurm_logs/%x_%j.err

set -euo pipefail

# 创建日志目录
mkdir -p exp/slurm_logs

echo "========== Job started at $(date) =========="


# 进入项目目录
cd /home/wh624/openpi-VTLA

# 设置环境变量
export LD_LIBRARY_PATH=/home/wh624/.lico_env/jupyterlab/env/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "Running training..."

# 运行训练
uv run scripts/train_tacrl_offline.py \
    uf850_pi05_lora_tactile_rl \
    --exp-name=test_tube_0409 \
    --critic_warmup_steps=500 \
    --q_weight=0.1 \
    --cql_alpha=0.01 \
    --cost_cql_alpha=0.01 \
    --use_safety_critic=True \
    --use_lagrangian=True \
    --cost_limit=4 \
    --tactile_cost_source=tactile \
    --tactile_cost_force_weight=1.0 \
    --tactile_cost_slip_weight=3.0 \
    --tactile_cost_asym_weight=0.1 \
    --tactile_cost_area_weight=100.0

echo "========== Job finished at $(date) =========="
