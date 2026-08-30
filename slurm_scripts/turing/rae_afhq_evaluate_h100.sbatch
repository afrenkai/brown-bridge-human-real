#!/bin/bash -l
#SBATCH --job-name=rae-afhq-evaluate
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --gres=gpu:H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
module load cuda

REPO_ROOT="/home/afrenk/brown-bridge-human-real"
RAE_ROOT="/home/afrenk/RAE"
PYTHON="${REPO_ROOT}/.venv-rae-managed/bin/python"
CONFIG="${REPO_ROOT}/outputs/rae_afhq_dit/afhq_cat_dog_dit_s_full/config.yaml"
GENERATED="${REPO_ROOT}/outputs/rae_afhq_generator/final_cfg_1.5_ode21"
TRAIN="${REPO_ROOT}/data/afhq_cat_dog_train"
VALIDATION="${REPO_ROOT}/data/afhq_cat_dog_validation"
OUTPUT="${REPO_ROOT}/outputs/rae_afhq_generator/evaluation_cfg_1.5_ode21"

cd "${RAE_ROOT}"
export PYTHONPATH="${RAE_ROOT}/src:${REPO_ROOT}/src"
export HF_HOME="/scratch/afrenk/huggingface"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

rm -rf "${OUTPUT}"
"${PYTHON}" "${REPO_ROOT}/experiments/evaluate_rae_afhq_generator.py" \
    --config "${CONFIG}" \
    --generated-root "${GENERATED}" \
    --train-root "${TRAIN}" \
    --validation-root "${VALIDATION}" \
    --output-dir "${OUTPUT}" \
    --batch-size 64 \
    --num-workers 12 \
    --probe-steps 500 \
    --seed 20260830
