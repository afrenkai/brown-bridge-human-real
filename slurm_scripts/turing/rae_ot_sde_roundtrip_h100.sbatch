#!/bin/bash -l
# Submit from any directory with:
# module load slurm && sbatch /home/afrenk/brown-bridge-human-real/slurm_scripts/turing/rae_ot_sde_roundtrip_h100.sbatch
#SBATCH --job-name=rae-ot-sde-gif
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --gres=gpu:H100:1
#SBATCH --constraint=H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

module load cuda

REPO_ROOT="/home/afrenk/brown-bridge-human-real"
RAE_ROOT="/home/afrenk/RAE"
AFHQ_ROOT="/home/afrenk/twisted-maps/data/afhq"
OUTPUT_ROOT="${REPO_ROOT}/outputs/rae_ot_sde_roundtrip"

cd "${RAE_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${RAE_ROOT}/src:${REPO_ROOT}/src"
export HF_HOME="/scratch/afrenk/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "RAE OT-SDE round-trip render"
echo "  brown-bridge revision: $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  RAE revision: $(git -C "${RAE_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  host: $(hostname)"
echo "  output: ${OUTPUT_ROOT}"

exec "${REPO_ROOT}/.venv-rae-managed/bin/python" \
    "${REPO_ROOT}/experiments/render_rae_ot_sde_roundtrip.py" \
    --rae-config "${RAE_ROOT}/configs/stage1/pretrained/DINOv2-B.yaml" \
    --afhq-root "${AFHQ_ROOT}" \
    --output-dir "${OUTPUT_ROOT}" \
    --split val \
    --pool-size 32 \
    --encode-batch-size 8 \
    --decode-batch-size 4 \
    --ot-epsilon 0.05 \
    --bridge-epsilon 0.002 \
    --half-steps 20 \
    --microsteps 4 \
    --frame-duration-ms 120 \
    --seed 20260715
