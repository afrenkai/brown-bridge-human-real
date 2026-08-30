#!/bin/bash -l
#SBATCH --job-name=rae-afhq-sample-pilot
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --gres=gpu:H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/home/afrenk/brown-bridge-human-real/logs/%x_%j.out
#SBATCH --error=/home/afrenk/brown-bridge-human-real/logs/%x_%j.err

set -euo pipefail

module load cuda

REPO_ROOT="/home/afrenk/brown-bridge-human-real"
RAE_ROOT="/home/afrenk/RAE"
PYTHON="${REPO_ROOT}/.venv-rae-managed/bin/python"
CONFIG="${REPO_ROOT}/outputs/rae_afhq_dit/afhq_cat_dog_dit_s_full/config.yaml"
CHECKPOINT="${REPO_ROOT}/outputs/rae_afhq_dit/afhq_cat_dog_dit_s_full/checkpoints/ep-last.pt"
OUTPUT_ROOT="${REPO_ROOT}/outputs/rae_afhq_generator/pilot"

cd "${RAE_ROOT}"
export PYTHONPATH="${RAE_ROOT}/src:${REPO_ROOT}/src"
export HF_HOME="/scratch/afrenk/huggingface"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for CFG_SCALE in 1.0 1.5 2.0; do
    EXTRA_ARGS=(--skip-trajectories)
    if [[ "${CFG_SCALE}" == "1.5" ]]; then
        EXTRA_ARGS=()
    fi
    "${PYTHON}" "${REPO_ROOT}/experiments/sample_rae_afhq_generator.py" \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${OUTPUT_ROOT}/cfg_${CFG_SCALE}" \
        --num-per-class 16 \
        --batch-size 8 \
        --seed 20260830 \
        --cfg-scale "${CFG_SCALE}" \
        --ode-steps 21 \
        --sde-steps 41 \
        "${EXTRA_ARGS[@]}"
done
