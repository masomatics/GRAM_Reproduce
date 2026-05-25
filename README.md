# GRAM_Reproduce

Reproduction of GRAM (Generative Recursive reAsoning Models, arXiv:2605.19376v2)
on the Sudoku-Extreme benchmark, built on top of TinyRecursiveModels (TRM).

## Layout

| Dir | Contents |
| --- | --- |
| `source/` | All Python code to read. Forked from `github.com/SamsungSAILMontreal/TinyRecursiveModels` at commit `c0110373` (MIT). Our GRAM additions live inline: `source/models/recursive_reasoning/gram.py`, `source/models/gram_loss.py`, `source/config/arch/gram.yaml`. |
| `jobs/` | PBS submission scripts (`*.pbs`) and bash wrappers (`*.sh`) for Miyabi. Includes `gram_train_long.pbs` (paper recipe), `gram_train_anticollapse.pbs` (anti-collapse variant), smoke tests, env setup. |
| `scripts/` | Python helper scripts: `count_params.py`, `dump_wandb_history.py`, `smoke_test.py`, `extract_pdf.py`. |
| `documents/` | Paper PDF. |
| `data/` | (gitignored) Symlinks to Sudoku-Extreme dataset prepared in `fast-slow-learning`. |
| `checkpoints/` | (gitignored) Model checkpoints written by training runs. |
| `wandb/`, `wandb_stdout/` | (gitignored) Local wandb logs and PBS stdout. |
| `env_extra/` | (gitignored) Python packages layered on `fast-slow-learning/env_akorn`. |
| `log_claude.txt` | Running session notes. |

## GRAM additions (vs. TRM)

| File | Role |
| --- | --- |
| `source/models/recursive_reasoning/gram.py` | `GenerativeRecursiveModel_ACTV1`: wraps TRM's recursive core, adds prior `p_θ(ε|u)` and posterior `q_φ(ε|u,y)` heads (SwiGLU MLPs that share the core), reparameterized ε sample, additive `z_t = u_t + ε_t`, emits both stop-grad KL halves. |
| `source/models/gram_loss.py` | `GRAMLossHead`: ELBO = CE + β·KL_balanced, with `kl_balance` (DreamerV2-style) and optional `free_bits`. |
| `source/config/arch/gram.yaml` | Hydra arch config: `halt_max_steps=16` (= N_sup), shared core hyperparams as TRM, GRAM head sizing. |

## Run

Smoke test (env, model, data path):
```
qsub /work/gj26/b20090/GRAM_Reproduce/jobs/smoke_test.pbs
qsub /work/gj26/b20090/GRAM_Reproduce/jobs/gram_smoke.pbs
```

Full training (paper recipe, ~3h on a single GH200):
```
qsub /work/gj26/b20090/GRAM_Reproduce/jobs/gram_train_long.pbs
```

Sync wandb (offline → cloud) from the login node:
```
module load python/3.10.16 && source /work/gj26/b20090/fast-slow-learning/env_akorn/bin/activate
wandb sync /work/gj26/b20090/GRAM_Reproduce/wandb/wandb/offline-run-*
```
