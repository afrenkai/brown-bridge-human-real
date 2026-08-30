#!/bin/bash -l
#SBATCH --job-name=setup-rae-python
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/afrenk/brown-bridge-human-real/logs/%x_%j.out
#SBATCH --error=/home/afrenk/brown-bridge-human-real/logs/%x_%j.err

set -euo pipefail

module load uv

REPO_ROOT="/home/afrenk/brown-bridge-human-real"
RAE_ROOT="/home/afrenk/RAE"
VENV="${REPO_ROOT}/.venv-rae-managed"

uv python install --managed-python 3.12
MANAGED_PYTHON=$(uv python find --managed-python 3.12)
if [[ ! -x "${VENV}/bin/python" ]]; then
    uv venv --python "${MANAGED_PYTHON}" "${VENV}"
fi
uv pip install --python "${VENV}/bin/python" \
    torch==2.8.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "${VENV}/bin/python" -r "${RAE_ROOT}/requirements.txt"
uv pip install --python "${VENV}/bin/python" --no-deps -e "${REPO_ROOT}"
uv pip install --python "${VENV}/bin/python" pillow huggingface_hub

"${VENV}/bin/python" - <<'PY'
from pathlib import Path
import sysconfig
import torch
include = Path(sysconfig.get_paths()['include'])
header = include / 'Python.h'
if not header.is_file():
    raise FileNotFoundError(header)
print(f'python_include={include}')
print(f'python_header={header}')
print(f'torch={torch.__version__}; cuda={torch.version.cuda}')
PY
