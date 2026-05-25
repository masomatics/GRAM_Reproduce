"""GRAM ELBO loss head.

Per supervision step t, computes
    L_t = CE(y, logits_t) + beta * KL_balanced(q_phi_t || p_theta_t)
with KL-balance (DreamerV2-style):
    KL_balanced = alpha * KL(stopgrad(q) || p) + (1 - alpha) * KL(q || stopgrad(p))
where alpha = kl_balance (paper: 0.8 for Sudoku-Extreme, beta=0.1).
The model emits both stop-grad halves; this head combines them.
"""
from typing import Any, Tuple, Dict, Sequence, Optional
import torch
import torch.nn.functional as F
from torch import nn

from models.losses import stablemax_cross_entropy, softmax_cross_entropy, IGNORE_LABEL_ID


class GRAMLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str,
                 beta_kl: float = 0.1, kl_balance: float = 0.8,
                 free_bits: float = 0.0, beta_warmup_steps: int = 0):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.beta_kl = float(beta_kl)
        self.kl_balance = float(kl_balance)
        self.free_bits = float(free_bits)
        # Linear KL warmup. effective_beta(step) = beta_kl * min(1, step/warmup_steps).
        # Lets the posterior learn an informative signal before the KL penalty bites.
        self.beta_warmup_steps = int(beta_warmup_steps)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(self, return_keys: Sequence[str], **model_kwargs
                ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        zero = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.float32)
        kl_q_stop = outputs.get("kl_q_stop", zero)
        kl_p_stop = outputs.get("kl_p_stop", zero)

        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            valid = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid.sum(),
                "accuracy": torch.where(valid, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid & seq_is_correct).sum(),
                "steps": torch.where(valid, new_carry.steps, 0).sum(),
            }

        # CE per-token, masked, averaged within each sequence then summed (matches TRM).
        ce_per_seq = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum(dim=-1)
        lm_loss = ce_per_seq.sum()

        # KL-balance: alpha * KL(stopgrad(q) || p) + (1 - alpha) * KL(q || stopgrad(p)).
        kl_balanced = self.kl_balance * kl_q_stop + (1.0 - self.kl_balance) * kl_p_stop
        if self.free_bits > 0:
            kl_balanced = torch.clamp_min(kl_balanced, self.free_bits)
        kl_loss = kl_balanced.sum()

        if self.training and self.beta_warmup_steps > 0:
            self._step += 1
            beta_eff = self.beta_kl * min(1.0, float(self._step.item()) / self.beta_warmup_steps)
        else:
            beta_eff = self.beta_kl
        total_loss = lm_loss + beta_eff * kl_loss

        # TRM's pretrain loop divides non-loss metric values by count (= valid.sum()).
        # Pre-multiply scalar quantities by count so the post-division yields the
        # actual value rather than value/count.
        cnt = metrics["count"].clamp_min(1)
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "kl_mean": (kl_balanced.mean() * cnt).detach(),
            "beta_eff": (torch.tensor(beta_eff, device=lm_loss.device) * cnt).detach(),
        })

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
