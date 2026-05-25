"""GRAM: Generative Recursive reAsoning Models.

Extends TRM's recursive core with a stochastic latent eps_t injected
between supervision steps, trained with amortized variational inference.

Per the paper (arXiv:2605.19376v2):
    eps_t ~ p_theta(eps_t | u_t) = N(mu_theta(u_t), sigma_theta^2(u_t) I)
    eps_t ~ q_phi(eps_t | u_t, y) = N(mu_phi(u_t, y), sigma_phi^2(u_t, y) I)
    z_t = u_t + eps_t
The recursive core (TRM's `L_level` block stack) is *shared* between
prior and posterior — only the small (mu, sigma) heads differ.

For Sudoku-Extreme the recursive core uses mlp_t=True
(SwiGLU+SwiGLU, no attention) per Section D.5.
"""
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
    TinyRecursiveReasoningModel_ACTV1Config,
)


@dataclass
class GenerativeRecursiveModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class GenerativeRecursiveModel_ACTV1Carry:
    inner_carry: GenerativeRecursiveModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class GenerativeRecursiveModel_ACTV1Config(TinyRecursiveReasoningModel_ACTV1Config):
    # GRAM-specific
    eps_expansion: float = 1.0          # SwiGLU expansion for mu/sigma heads
    eps_logsigma_init: float = -2.0     # initial bias of log-sigma head -> sigma ~ 0.13 at start


def _gaussian_kl(mu_q: torch.Tensor, logsigma_q: torch.Tensor,
                 mu_p: torch.Tensor, logsigma_p: torch.Tensor) -> torch.Tensor:
    """KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)) per element, in float32."""
    mu_q = mu_q.float(); logsigma_q = logsigma_q.float()
    mu_p = mu_p.float(); logsigma_p = logsigma_p.float()
    var_q = (2 * logsigma_q).exp()
    var_p = (2 * logsigma_p).exp()
    return logsigma_p - logsigma_q + 0.5 * ((var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)


class _MuSigmaHead(nn.Module):
    """Two SwiGLU MLPs producing (mu, log_sigma) of shape [..., D].

    Used both for the prior (input=u_t) and the posterior (input=concat(u_t, y_emb) -> D).
    The posterior caller passes pre-concatenated/-projected input of dim D.
    """
    def __init__(self, hidden_size: int, expansion: float, logsigma_init: float):
        super().__init__()
        self.mu = SwiGLU(hidden_size=hidden_size, expansion=expansion)
        self.logsigma = SwiGLU(hidden_size=hidden_size, expansion=expansion)
        # Use default trunc-normal init for logsigma's down_proj (NOT zeroed) — zeroing
        # made gradients into sigma vanish at init and locked the model into KL=0.
        self.logsigma_init = nn.Parameter(torch.full((hidden_size,), float(logsigma_init)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(x)
        logsigma = self.logsigma(x) + self.logsigma_init
        return mu, logsigma


class GenerativeRecursiveModel_ACTV1_Inner(nn.Module):
    """TRM inner + stochastic latent eps_t injected on z_H."""

    def __init__(self, config: GenerativeRecursiveModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size,
                                            init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = (-(self.config.puzzle_emb_ndim // -self.config.hidden_size)
                               if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len)
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers,
                                                    self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size,
                                                    init_std=0, cast_to=self.forward_dtype)

        if self.config.pos_encodings == "rope":
            from models.layers import RotaryEmbedding
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len,
                                             self.config.hidden_size, init_std=embed_init_std,
                                             cast_to=self.forward_dtype)

        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _ in range(self.config.L_layers)]
        )

        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

        # GRAM: stochastic latent heads. Prior takes u_t directly.
        # Posterior takes [u_t, y_emb] projected from 2D -> D.
        D = self.config.hidden_size
        self.prior_head = _MuSigmaHead(D, self.config.eps_expansion, self.config.eps_logsigma_init)
        self.posterior_in = CastedLinear(2 * D, D, bias=False)
        self.posterior_head = _MuSigmaHead(D, self.config.eps_expansion, self.config.eps_logsigma_init)

        # LPRM (Latent Process Reward Model): v_psi reads the first token of z_H_stoch.
        # Trained as MSE against per-step correctness; gives eps a "predict the answer
        # quality" job so it carries information beyond what CE alone needs.
        self.v_head = CastedLinear(D, 1, bias=True)
        with torch.no_grad():
            self.v_head.weight.zero_()
            self.v_head.bias.zero_()  # type: ignore

    def _input_embeddings(self, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(inputs.to(torch.int32))
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * embedding

    def _label_embeddings(self, labels: torch.Tensor) -> torch.Tensor:
        # Match shape of z_H: [B, puzzle_emb_len + seq_len, D]. Pad puzzle_emb_len positions with zeros.
        emb = self.embed_tokens(labels.clamp_min(0).to(torch.int32))  # IGNORE_LABEL_ID is -100 in dataset; clamp_min for safety
        if self.puzzle_emb_len > 0:
            pad = torch.zeros(emb.shape[0], self.puzzle_emb_len, emb.shape[-1], dtype=emb.dtype, device=emb.device)
            emb = torch.cat((pad, emb), dim=-2)
        return self.embed_scale * emb

    def empty_carry(self, batch_size: int):
        return GenerativeRecursiveModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: GenerativeRecursiveModel_ACTV1InnerCarry):
        return GenerativeRecursiveModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: GenerativeRecursiveModel_ACTV1InnerCarry,
                batch: Dict[str, torch.Tensor],
                sample_posterior: bool) -> Tuple[GenerativeRecursiveModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Deterministic transition: produce u_t = (z_H, z_L).
        z_H, z_L = carry.z_H, carry.z_L
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        u_H = z_H  # high-level deterministic update at this supervision step

        # Stochastic latent: prior always, posterior only at training.
        mu_p, logsigma_p = self.prior_head(u_H)

        if sample_posterior:
            y_emb = self._label_embeddings(batch["labels"])
            post_in = self.posterior_in(torch.cat([u_H, y_emb], dim=-1))
            mu_q, logsigma_q = self.posterior_head(post_in)
            eps = mu_q + (logsigma_q.exp()) * torch.randn_like(mu_q)
            # DreamerV2-style KL balance: emit both stop-grad halves separately;
            # the loss head combines them with alpha = kl_balance.
            kl_q_stop = _gaussian_kl(mu_q.detach(), logsigma_q.detach(), mu_p, logsigma_p)
            kl_p_stop = _gaussian_kl(mu_q, logsigma_q, mu_p.detach(), logsigma_p.detach())
            kl_q_stop = kl_q_stop.mean(dim=(-1, -2))
            kl_p_stop = kl_p_stop.mean(dim=(-1, -2))
        else:
            eps = mu_p + (logsigma_p.exp()) * torch.randn_like(mu_p)
            zero = torch.zeros(u_H.shape[0], device=u_H.device, dtype=torch.float32)
            kl_q_stop = zero
            kl_p_stop = zero

        z_H_stoch = u_H + eps.to(u_H.dtype)

        # New (detached) carry for next supervision step
        new_carry = GenerativeRecursiveModel_ACTV1InnerCarry(z_H=z_H_stoch.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H_stoch)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H_stoch[:, 0]).to(torch.float32)
        # LPRM: v_psi reads the first token of z_H_stoch; returns scalar reward prediction per sample.
        v_pred = self.v_head(z_H_stoch[:, 0]).to(torch.float32).squeeze(-1)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), (kl_q_stop, kl_p_stop), v_pred


class GenerativeRecursiveModel_ACTV1(nn.Module):
    """ACT wrapper for GRAM. Mirrors TRM's wrapper; passes train-flag to inner for posterior sampling."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = GenerativeRecursiveModel_ACTV1Config(**config_dict)
        self.inner = GenerativeRecursiveModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return GenerativeRecursiveModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(self, carry: GenerativeRecursiveModel_ACTV1Carry, batch: Dict[str, torch.Tensor]):
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
                            for k, v in carry.current_data.items()}

        new_inner_carry, logits, (q_halt_logits, q_continue_logits), (kl_q_stop, kl_p_stop), v_pred = self.inner(
            new_inner_carry, new_current_data, sample_posterior=self.training,
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "kl_q_stop": kl_q_stop,
            "kl_p_stop": kl_p_stop,
            "v_pred": v_pred,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            # GRAM uses fixed N_sup supervision steps; halt only at max.
            halted = is_last_step

        return GenerativeRecursiveModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
