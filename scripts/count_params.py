"""Print GRAM and TRM Sudoku parameter counts to compare with paper (10.9M)."""
import sys
sys.path.insert(0, "/work/gj26/b20090/GRAM_Reproduce/source")

import torch

# Common Sudoku-Extreme arch settings (paper + TRM defaults)
common = dict(
    batch_size=64,
    seq_len=81,
    puzzle_emb_ndim=512,
    num_puzzle_identifiers=1,
    vocab_size=11,
    H_cycles=3, L_cycles=6,
    H_layers=0, L_layers=2,
    hidden_size=512,
    expansion=4.0,
    num_heads=8,
    pos_encodings="none",
    halt_max_steps=16,
    halt_exploration_prob=0.0,
    forward_dtype="bfloat16",
    mlp_t=True,             # Sudoku special case
    puzzle_emb_len=16,
    no_ACT_continue=True,
)


def count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fmt(n: int) -> str:
    return f"{n:,} ({n/1e6:.2f}M)"


def print_breakdown(name: str, model: torch.nn.Module):
    print(f"\n=== {name}: total trainable = {fmt(count(model))} ===")
    for child_name, child in model.named_children():
        n = count(child)
        if n > 0:
            print(f"  {child_name}: {fmt(n)}")
            for sub_name, sub in child.named_children():
                if count(sub) > 0:
                    print(f"    {sub_name}: {fmt(count(sub))}")


with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    trm = TinyRecursiveReasoningModel_ACTV1({**common, "causal": False})
    print_breakdown("TRM (mlp_t=True, Sudoku)", trm)

    for exp in (1.0, 2.0, 4.0):
        from models.recursive_reasoning.gram import GenerativeRecursiveModel_ACTV1
        gram = GenerativeRecursiveModel_ACTV1({**common, "causal": False,
                                               "eps_expansion": exp, "eps_logsigma_init": -2.0})
        print_breakdown(f"GRAM (eps_expansion={exp})", gram)
        del gram
