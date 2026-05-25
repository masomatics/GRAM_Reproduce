#!/bin/bash
# Periodic wandb sync + log append. Designed to run as a persistent Monitor.
# Each iteration emits one notification line per active wandb run AND appends
# the same line to log_claude.txt.

set -u
GRAM=/work/gj26/b20090/GRAM_Reproduce
LOG="$GRAM/log_claude.txt"
SYNC_EVERY_SECS="${SYNC_EVERY_SECS:-900}"   # 15 min default

module load python/3.10.16 2>/dev/null
# shellcheck disable=SC1091
source /work/gj26/b20090/fast-slow-learning/env_akorn/bin/activate

# Print a header so the log makes sense
{ echo ""; echo "[$(date -Is)] auto_sync_log started (every ${SYNC_EVERY_SECS}s)"; } >> "$LOG"

while true; do
    # Find offline-run directories whose .wandb file was modified in last 2h.
    ACTIVE_DIRS=$(find "$GRAM/wandb/wandb" -maxdepth 2 -name "run-*.wandb" -mmin -120 -print 2>/dev/null \
                  | xargs -I{} dirname {} 2>/dev/null | sort -u)
    if [ -z "$ACTIVE_DIRS" ]; then
        echo "[$(date -Is)] no active wandb runs in last 2h"
        sleep "$SYNC_EVERY_SECS"
        continue
    fi

    for d in $ACTIVE_DIRS; do
        wandb sync "$d" >/dev/null 2>&1 || true
        LATEST="${d##*/}"
        RUN_ID="${LATEST##*-}"

        SUMMARY=$(python - <<PY 2>/dev/null
import wandb
api = wandb.Api()
try:
    r = api.run("koyama-masanori-masomatics/Sudoku-extreme-1k-aug-1000-ACT-torch/${RUN_ID}")
except Exception as e:
    print(f"FAIL {e}")
else:
    a = dict(r.summary.get("all", {}) or {})
    print(
        f"step={r.summary.get('_step')}",
        f"train/lm_loss={r.summary.get('train/lm_loss'):.3g}" if r.summary.get('train/lm_loss') is not None else "train/lm_loss=-",
        f"train/kl_mean={r.summary.get('train/kl_mean'):.3g}" if r.summary.get('train/kl_mean') is not None else "train/kl_mean=-",
        f"beta_eff={r.summary.get('train/beta_eff'):.3g}" if r.summary.get('train/beta_eff') is not None else "beta_eff=-",
        f"eval/acc={a.get('accuracy', 0):.3f}",
        f"eval/exact={a.get('exact_accuracy', 0):.3f}",
        f"eval/lm_loss={a.get('lm_loss', 0):.3f}" if a else "eval/lm_loss=-",
    )
PY
)

        STAMP=$(date -Is)
        LINE="[$STAMP] run=${RUN_ID} ${SUMMARY}"
        echo "$LINE"                 # event line -> chat notification
        echo "$LINE" >> "$LOG"       # persistent file record
    done

    sleep "$SYNC_EVERY_SECS"
done
