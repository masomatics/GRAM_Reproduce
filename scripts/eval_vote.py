"""Multi-sample voting eval for GRAM checkpoints.

For each test puzzle, draws K independent eps trajectories from the learned
prior (each runs N_sup supervision steps), then aggregates the K candidate
grids via either:
  - majority : per-cell majority vote across K argmax decodes
  - best_of_N: pick the single trajectory whose mean LPRM v_psi(z_t) is max

Paper (arXiv:2605.19376v2 Sec 2.3 / 4.1): "GRAM with N=20 samples at 16
iterations outperforms all deterministic baselines... To select among
candidates, we use either majority voting or best-of-N with a Latent
Process Reward Model (LPRM)."

Usage:
  python scripts/eval_vote.py \\
    --checkpoint checkpoints/gram_sudoku_anticollapse/step_65100 \\
    --data data/sudoku-extreme-1k-aug-1000 \\
    --K 20 --strategy majority --batch_size 768 --max_batches 0
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

# Make source/ importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "source"))

from models.recursive_reasoning.gram import (
    GenerativeRecursiveModel_ACTV1,
)


def build_model(meta: dict, halt_max_steps: int = 16, batch_size: int = 768) -> torch.nn.Module:
    cfg = dict(
        batch_size=batch_size,
        seq_len=int(meta["seq_len"]),
        vocab_size=int(meta["vocab_size"]),
        num_puzzle_identifiers=int(meta["num_puzzle_identifiers"]),

        H_cycles=3, L_cycles=6,
        H_layers=0, L_layers=2,
        hidden_size=512, num_heads=8, expansion=4,
        puzzle_emb_ndim=512, puzzle_emb_len=16,
        pos_encodings="none", forward_dtype="bfloat16",
        mlp_t=True,
        no_ACT_continue=True,
        halt_exploration_prob=0.0,
        halt_max_steps=halt_max_steps,
        rope_theta=10000.0,

        eps_expansion=2.0,
        eps_logsigma_init=-2.0,
    )
    return GenerativeRecursiveModel_ACTV1(cfg)


def load_state(model: torch.nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cuda")
    # Strip optional "_orig_mod.model." prefix added by torch.compile + loss-head wrap.
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for pref in ("_orig_mod.model.", "_orig_mod.", "model."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
                break
        new_sd[nk] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False, assign=True)
    print(f"  loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("    first missing:", missing[:5])
    if unexpected:
        print("    first unexpected:", unexpected[:5])


def iter_test_batches(data_root: str, batch_size: int):
    """Stream test batches as contiguous slices over the test arrays."""
    test_dir = os.path.join(data_root, "test")
    inputs = np.load(os.path.join(test_dir, "all__inputs.npy"), mmap_mode="r")
    labels = np.load(os.path.join(test_dir, "all__labels.npy"), mmap_mode="r")
    pids = np.load(os.path.join(test_dir, "all__puzzle_identifiers.npy"), mmap_mode="r")
    n = inputs.shape[0]
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        yield (
            torch.from_numpy(np.ascontiguousarray(inputs[s:e])).cuda(),
            torch.from_numpy(np.ascontiguousarray(labels[s:e])).cuda(),
            torch.from_numpy(np.ascontiguousarray(pids[s:e])).cuda(),
        )


@torch.inference_mode()
def run_one_trajectory(model, batch, N_sup: int):
    """Run a single K=1 rollout for the batch; returns (logits_final [B,T,V], v_mean [B])."""
    carry = model.initial_carry(batch)
    v_accum = torch.zeros(batch["inputs"].shape[0], dtype=torch.float32, device="cuda")
    logits_final = None
    for step in range(N_sup):
        carry, outputs = model(carry=carry, batch=batch)
        v_accum += outputs["v_pred"].float()
        if step == N_sup - 1:
            logits_final = outputs["logits"]
    return logits_final, v_accum / N_sup


@torch.inference_mode()
def aggregate(preds_K: torch.Tensor, scores_K: torch.Tensor, strategy: str) -> torch.Tensor:
    """preds_K: [K, B, T] argmax token ids. scores_K: [K, B] LPRM mean. Return [B, T]."""
    K, B, T = preds_K.shape
    if strategy == "majority":
        # Per-cell mode across K. Vocab=11 -> small; do scatter-add bincount.
        V = int(preds_K.max().item()) + 1
        votes = torch.zeros(B, T, V, dtype=torch.int32, device=preds_K.device)
        votes.scatter_add_(2, preds_K.permute(1, 2, 0), torch.ones_like(preds_K.permute(1, 2, 0), dtype=torch.int32))
        return votes.argmax(dim=-1)
    elif strategy == "best_of_N":
        # Pick the trajectory whose mean LPRM score is highest per puzzle.
        best_k = scores_K.argmax(dim=0)  # [B]
        idx = best_k.view(1, B, 1).expand(1, B, T)
        return preds_K.gather(0, idx).squeeze(0)
    elif strategy == "k1":
        return preds_K[0]
    else:
        raise ValueError(strategy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/sudoku-extreme-1k-aug-1000")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--N_sup", type=int, default=16)
    ap.add_argument("--strategy", choices=["majority", "best_of_N", "k1"], default="majority")
    ap.add_argument("--batch_size", type=int, default=768)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = full test set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta = json.load(open(os.path.join(args.data, "test", "dataset.json")))
    model = build_model(meta, halt_max_steps=args.N_sup, batch_size=args.batch_size).cuda()
    load_state(model, args.checkpoint)
    model.eval()
    print(f"  K={args.K}  strategy={args.strategy}  N_sup={args.N_sup}  batch={args.batch_size}")

    total_seqs = 0
    correct_tokens = 0
    total_tokens = 0
    exact_seqs = 0
    t0 = time.time()

    for bi, (inp, lab, pid) in enumerate(iter_test_batches(args.data, args.batch_size)):
        if args.max_batches and bi >= args.max_batches:
            break
        batch = {"inputs": inp, "labels": lab, "puzzle_identifiers": pid}

        preds_K = []
        scores_K = []
        for k in range(args.K):
            logits, v_mean = run_one_trajectory(model, batch, args.N_sup)
            preds_K.append(logits.argmax(dim=-1))  # [B, T]
            scores_K.append(v_mean)
        preds_K_t = torch.stack(preds_K, dim=0)    # [K, B, T]
        scores_K_t = torch.stack(scores_K, dim=0)  # [K, B]

        voted = aggregate(preds_K_t, scores_K_t, args.strategy)  # [B, T]

        # Ignore -100 padded labels (IGNORE_LABEL_ID handled in losses.py); dataset uses 0 as pad.
        mask = (lab != 0)
        is_correct = (voted == lab) & mask
        per_seq_tok = mask.sum(-1).clamp_min(1)
        per_seq_correct = is_correct.sum(-1)
        seq_exact = per_seq_correct == mask.sum(-1)

        bs = inp.shape[0]
        total_seqs += bs
        correct_tokens += per_seq_correct.sum().item()
        total_tokens += mask.sum().item()
        exact_seqs += seq_exact.sum().item()

        if bi % 5 == 0:
            elap = time.time() - t0
            print(f"  batch {bi:4d}  seqs={total_seqs:7d}  "
                  f"per_tok={correct_tokens/max(total_tokens,1):.4f}  "
                  f"exact={exact_seqs/max(total_seqs,1):.4f}  "
                  f"elap={elap:.1f}s", flush=True)

    elap = time.time() - t0
    per_tok = correct_tokens / max(total_tokens, 1)
    exact = exact_seqs / max(total_seqs, 1)
    print()
    print(f"FINAL  seqs={total_seqs}  K={args.K}  strategy={args.strategy}")
    print(f"  per_token_accuracy = {per_tok:.4f}")
    print(f"  exact_accuracy     = {exact:.4f}")
    print(f"  elapsed            = {elap:.1f}s")


if __name__ == "__main__":
    main()
