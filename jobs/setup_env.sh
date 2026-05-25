#!/bin/bash
# Set up GRAM_Reproduce env_extra layered on top of fast-slow-learning/env_akorn.
# Sourced/executed from inside a GPU job (debug-g) so nvcc + GPU exist for adam-atan2.

set -eu

GRAM=/work/gj26/b20090/GRAM_Reproduce
AKORN_VENV=/work/gj26/b20090/fast-slow-learning/env_akorn
EXTRA="$GRAM/env_extra"

module purge
module load python/3.10.16
module load cuda/12.6
module load cudnn/9.5.1.17

# shellcheck disable=SC1091
source "$AKORN_VENV/bin/activate"

echo "==> python: $(which python) $(python --version)"
echo "==> torch:  $(python -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "==> nvcc:   $(nvcc --version | head -4 | tail -1)"

mkdir -p "$EXTRA"
export PYTHONPATH="$EXTRA${PYTHONPATH:+:$PYTHONPATH}"

# Stage 1: pure-python deps
pip install --no-cache-dir --target "$EXTRA" \
  hydra-core==1.3.2 omegaconf==2.3.0 coolname==2.2.0 \
  numba==0.61.2 ninja

# Stage 2: adam-atan2 (CUDA extension, built against installed torch)
pip install --no-cache-dir --no-build-isolation --target "$EXTRA" adam-atan2==0.0.3

# Smoke imports
python - <<'PY'
import sys
print("PYTHONPATH:", sys.path[:3])
import torch, einops, hydra, omegaconf, coolname, numba, triton, argdantic, pydantic
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("einops", einops.__version__)
print("hydra", hydra.__version__)
print("omegaconf", omegaconf.__version__)
print("coolname", coolname.__version__)
print("numba", numba.__version__)
print("triton", triton.__version__)
from adam_atan2 import AdamATan2
print("AdamATan2 ok:", AdamATan2)
PY
