#!/bin/bash -l
#SBATCH --job-name=rae-afhq-generate-evalset
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --gres=gpu:H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/home/afrenk/brown-bridge-human-real/logs/%x_%j.out
#SBATCH --error=/home/afrenk/brown-bridge-human-real/logs/%x_%j.err

set -euo pipefail
module load cuda

REPO_ROOT="/home/afrenk/brown-bridge-human-real"
RAE_ROOT="/home/afrenk/RAE"
PYTHON="${REPO_ROOT}/.venv-rae-managed/bin/python"
CONFIG="${REPO_ROOT}/outputs/rae_afhq_dit/afhq_cat_dog_dit_s_full/config.yaml"
CHECKPOINT="${REPO_ROOT}/outputs/rae_afhq_dit/afhq_cat_dog_dit_s_full/checkpoints/ep-last.pt"
OUTPUT="${REPO_ROOT}/outputs/rae_afhq_generator/final_cfg_1.5_ode21"

cd "${RAE_ROOT}"
export PYTHONPATH="${RAE_ROOT}/src:${REPO_ROOT}/src"
export HF_HOME="/scratch/afrenk/huggingface"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

rm -rf "${OUTPUT}"
"${PYTHON}" "${REPO_ROOT}/experiments/sample_rae_afhq_generator.py" \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT}" \
    --num-per-class 5000 \
    --batch-size 16 \
    --writer-workers 8 \
    --seed 20260830 \
    --cfg-scale 1.5 \
    --ode-steps 21 \
    --sde-steps 101
