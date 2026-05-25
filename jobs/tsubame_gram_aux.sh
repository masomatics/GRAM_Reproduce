#!/bin/bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=08:00:00
#$ -j y
#$ -N gram_aux
#$ -o $HOME/gram_aux.log

# Full GRAM training on TSUBAME with paper aux losses (anti-collapse + L_ACT + L_LPRM).
# node_q=1 = 1 full H100 (80GB), 8h walltime.

set -eu
module purge
module load apptainer-olderenv

GRAM=$HOME/GRAM_Reproduce
CKPT=$GRAM/checkpoints/gram_sudoku_aux
mkdir -p "$CKPT" "$GRAM/wandb"

apptainer exec --cleanenv --home "$HOME" \
  --env WANDB_API_KEY="${WANDB_API_KEY:-}" --env WANDB_MODE=offline \
  --env PYTHONPATH="$GRAM/env_extra" \
  --env WANDB_DIR="$GRAM/wandb" \
  --nv "$HOME/singularity/pytorch_25.01.sif" \
  /bin/bash -lc "
    set -eu
    cd $GRAM/source
    python pretrain.py \
      arch=gram \
      data_paths=\"[$GRAM/data/sudoku-extreme-1k-aug-1000]\" \
      evaluators=\"[]\" \
      epochs=50000 eval_interval=5000 \
      global_batch_size=768 \
      lr=1e-4 puzzle_emb_lr=1e-4 \
      weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
      arch.mlp_t=True arch.pos_encodings=none \
      arch.L_layers=2 \
      arch.H_cycles=3 arch.L_cycles=6 \
      arch.eps_logsigma_init=0.0 \
      arch.loss.beta_kl=0.1 arch.loss.kl_balance=0.8 \
      arch.loss.free_bits=0.05 arch.loss.beta_warmup_steps=10000 \
      arch.loss.act_weight=0.5 arch.loss.lprm_weight=0.5 \
      checkpoint_every_eval=True \
      +checkpoint_path=$CKPT \
      +run_name=gram_sudoku_aux_tsubame ema=True
  "
