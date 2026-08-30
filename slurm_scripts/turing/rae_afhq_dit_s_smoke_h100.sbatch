#!/bin/bash -l
#SBATCH --job-name=rae-afhq-dit-smoke
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --gres=gpu:H100:4
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --output=/home/afrenk/brown-bridge-human-real/logs/%x_%j.out
#SBATCH --error=/home/afrenk/brown-bridge-human-real/logs/%x_%j.err

set -euo pipefail

module load cuda

REPO_ROOT="/home/afrenk/brown-bridge-human-real"
RAE_ROOT="/home/afrenk/RAE"
PYTHON="${REPO_ROOT}/.venv-rae-managed/bin/python"
DATA_ROOT="${REPO_ROOT}/data/afhq_cat_dog_smoke"
RESULTS_ROOT="${REPO_ROOT}/outputs/rae_afhq_dit"
SMOKE_CONFIG="${RESULTS_ROOT}/smoke_config.yaml"

mkdir -p "${RESULTS_ROOT}"
cd "${RAE_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${RAE_ROOT}/src:${REPO_ROOT}/src"
export HF_HOME="/scratch/afrenk/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EXPERIMENT_NAME="afhq_cat_dog_dit_s_smoke"

"${PYTHON}" - <<'PY'
from omegaconf import OmegaConf
source = OmegaConf.load('/home/afrenk/brown-bridge-human-real/configs/rae_afhq_dit_s.yaml')
source.training.epochs = 1
source.training.global_batch_size = 32
source.training.num_workers = 8
source.training.checkpoint_interval = 1
source.training.sample_every = 1000
source.training.scheduler.warmup_epochs = 0
source.training.scheduler.decay_end_epoch = 1
OmegaConf.save(source, '/home/afrenk/brown-bridge-human-real/outputs/rae_afhq_dit/smoke_config.yaml')
PY

echo "RAE AFHQ DiT-S smoke fine-tune"
echo "  host: $(hostname)"
echo "  data: ${DATA_ROOT}"
echo "  results: ${RESULTS_ROOT}/${EXPERIMENT_NAME}"

exec "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    "${RAE_ROOT}/src/train.py" \
    --config "${SMOKE_CONFIG}" \
    --data-path "${DATA_ROOT}" \
    --results-dir "${RESULTS_ROOT}" \
    --image-size 256 \
    --precision bf16 \
    --compile
