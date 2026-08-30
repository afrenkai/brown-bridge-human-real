#!/bin/bash -l
#SBATCH --job-name=rae-afhq-dit-full
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --gres=gpu:H100:4
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=23:30:00
#SBATCH --output=/home/afrenk/brown-bridge-human-real/logs/%x_%j.out
#SBATCH --error=/home/afrenk/brown-bridge-human-real/logs/%x_%j.err

set -euo pipefail

module load cuda

REPO_ROOT="/home/afrenk/brown-bridge-human-real"
RAE_ROOT="/home/afrenk/RAE"
PYTHON="${REPO_ROOT}/.venv-rae-managed/bin/python"
DATA_ROOT="${REPO_ROOT}/data/afhq_cat_dog_train"
RESULTS_ROOT="${REPO_ROOT}/outputs/rae_afhq_dit"
FULL_CONFIG="${RESULTS_ROOT}/full_config.yaml"

mkdir -p "${RESULTS_ROOT}"
cd "${RAE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${RAE_ROOT}/src:${REPO_ROOT}/src"
export HF_HOME="/scratch/afrenk/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EXPERIMENT_NAME="afhq_cat_dog_dit_s_full"

"${PYTHON}" - <<'PY'
from omegaconf import OmegaConf
source = OmegaConf.load('/home/afrenk/brown-bridge-human-real/configs/rae_afhq_dit_s.yaml')
source.training.global_batch_size = 32
source.training.num_workers = 8
OmegaConf.save(source, '/home/afrenk/brown-bridge-human-real/outputs/rae_afhq_dit/full_config.yaml')
PY

echo "RAE AFHQ DiT-S full fine-tune"
echo "  host: $(hostname)"
echo "  data: ${DATA_ROOT}"
echo "  results: ${RESULTS_ROOT}/${EXPERIMENT_NAME}"

exec "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    "${RAE_ROOT}/src/train.py" \
    --config "${FULL_CONFIG}" \
    --data-path "${DATA_ROOT}" \
    --results-dir "${RESULTS_ROOT}" \
    --image-size 256 \
    --precision bf16 \
    --compile
