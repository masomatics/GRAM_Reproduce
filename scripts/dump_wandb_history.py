"""Fetch the current run's logged metrics from wandb and pretty-print."""
import wandb

api = wandb.Api()
run = api.run("koyama-masanori-masomatics/Sudoku-extreme-1k-aug-1000-ACT-torch/s957pb3l")
print(f"Run: {run.name}  state={run.state}  params={run.summary.get('num_params')}")
print(f"URL: {run.url}\n")

keys = [
    "train/lm_loss", "train/kl_loss", "train/kl_mean",
    "train/accuracy", "train/exact_accuracy",
    "train/q_halt_loss", "train/q_continue_loss",
    "train/steps", "train/count", "train/lr",
    "test/lm_loss", "test/accuracy", "test/exact_accuracy", "test/steps",
]
print("summary keys:", sorted(list(run.summary.keys())))
print("\nsummary['all']:", dict(run.summary.get("all", {})))

print("\nALL EVAL ROWS (via scan_history):")
for r in run.scan_history(keys=None):
    if "all" in r and isinstance(r["all"], dict):
        d = r["all"]
        print(f"  step={r.get('_step')}  acc={d.get('accuracy', 0):.4f}"
              f"  exact={d.get('exact_accuracy', 0):.4f}"
              f"  lm_loss={d.get('lm_loss', 0):.4f}"
              f"  kl={d.get('kl_loss', 0):.4f}  steps={d.get('steps', 0):.1f}")
hist = run.history(samples=2000, pandas=False, x_axis="_step")

if not hist:
    print("No history rows yet.")
else:
    rows = sorted(hist, key=lambda r: r["_step"])
    cols = ["_step", "train/lm_loss", "train/kl_loss", "train/kl_mean",
            "train/accuracy", "train/exact_accuracy",
            "all/lm_loss", "all/accuracy", "all/exact_accuracy", "all/steps"]
    header = " | ".join(f"{c:>20}" for c in cols)
    print(header); print("-" * len(header))
    # Print first 5, last 5, plus any rows containing test/*
    test_rows = [r for r in rows if "all" in r and isinstance(r["all"], dict)]
    print(f"(history rows: {len(rows)} total, {len(test_rows)} with all/* eval metrics)")
    print("\nEVAL METRICS:")
    for r in test_rows:
        d = r["all"]
        print(f"  step={r['_step']}  lm_loss={d.get('lm_loss', 0):.4f}"
              f"  accuracy={d.get('accuracy', 0):.4f}"
              f"  exact_accuracy={d.get('exact_accuracy', 0):.4f}"
              f"  steps={d.get('steps', 0):.1f}")
    print()
    sample = rows[:5] + test_rows + rows[-5:]
    seen = set()
    for r in sample:
        s = r.get("_step")
        if s in seen: continue
        seen.add(s)
        cells = []
        for c in cols:
            v = r.get(c)
            if v is None:
                cells.append(f"{'-':>20}")
            else:
                cells.append(f"{v:>20.4f}" if isinstance(v, float) else f"{str(v):>20}")
        print(" | ".join(cells))
