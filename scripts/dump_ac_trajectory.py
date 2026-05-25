"""Dump AC run trajectory: KL evolution from scan_history + each eval's all/* metrics."""
import wandb

api = wandb.Api()
r = api.run("koyama-masanori-masomatics/Sudoku-extreme-1k-aug-1000-ACT-torch/ljgq5pop")
print(f"state={r.state}  latest step={r.summary.get('_step')}\n")

print("=== Eval rows (each contains an 'all' nested dict) ===")
for row in r.scan_history():
    if "all" in row and isinstance(row["all"], dict):
        a = row["all"]
        print(f"  step={row.get('_step')}  acc={a.get('accuracy', 0):.4f}"
              f"  exact={a.get('exact_accuracy', 0):.4f}"
              f"  lm_loss={a.get('lm_loss', 0):.3f}"
              f"  kl_mean={a.get('kl_mean', 0):.4f}"
              f"  steps={a.get('steps', 0):.1f}")

print("\n=== KL trajectory (every 1000 steps approx) ===")
last_logged = -1
for row in r.scan_history(keys=["_step", "train/kl_mean", "train/beta_eff", "train/lm_loss"]):
    s = row.get("_step", 0)
    if s - last_logged >= 1000:
        last_logged = s
        kl = row.get("train/kl_mean")
        be = row.get("train/beta_eff")
        ll = row.get("train/lm_loss")
        print(f"  step={s:6d}  train/kl_mean={kl}  beta_eff={be}  lm_loss={ll}")
