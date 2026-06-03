#!/bin/bash
#$ -cwd
#$ -l node_o=1
#$ -l h_rt=00:30:00
#$ -j y
#$ -N gram_setup
#$ -o $HOME/gram_setup.log

# TSUBAME env setup. Installs hydra-core / omegaconf / coolname / adam-atan2 / numba
# into $HOME/GRAM_Reproduce/env_extra so pretrain.py can import them.

set -eu
module purge
module load apptainer-olderenv

GRAM=$HOME/GRAM_Reproduce
EXTRA=$GRAM/env_extra
mkdir -p "$EXTRA"

apptainer exec --cleanenv --home "$HOME" --nv "$HOME/singularity/pytorch_25.01.sif" \
  /bin/bash -lc "
    set -eu
    python --version
    python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())'
    # Wipe env_extra first to drop any prior numpy 2.x.
    find $EXTRA -mindepth 1 -delete 2>/dev/null || true
    pip install --no-cache-dir --target=$EXTRA \
      'numpy<2' \
      hydra-core==1.3.2 omegaconf==2.3.0 coolname==2.2.0 \
      numba==0.61.2 ninja einops==0.8.1 argdantic==1.3.3 \
      wandb==0.22.2 huggingface_hub==0.34.4
    pip install --no-cache-dir --no-build-isolation --target=$EXTRA adam-atan2==0.0.3
    PYTHONPATH=$EXTRA python - <<'PY'
import sys
print('PYTHONPATH head:', sys.path[:3])
import torch, einops, hydra, omegaconf, coolname, numba, argdantic, wandb
from adam_atan2 import AdamATan2
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())
print('einops', einops.__version__)
print('hydra', hydra.__version__)
print('wandb', wandb.__version__)
print('AdamATan2 ok')
PY
  "
