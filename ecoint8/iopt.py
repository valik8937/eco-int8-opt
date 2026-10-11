"""I8ECO: integer-only ECO step. No fp32 tensors anywhere in this file.

Working precision: per-channel fine units ``U_o = sw_o / 2**F`` (F=12),
int64 accumulators. Every real-valued factor (lr, beta, alpha, scale ratios)
becomes an integer ``(mult, shift)`` pair via ``quantize_multiplier`` —
scalar Python math only. Stochastic rounding via ``torch.randint``.

One step per weight row ``o`` (``U = sw/2**F``)::

    mU  = m_int * (s_m / U) + g_int * (s_g / U) * ... (momentum, transient int64)
    thU = (w_int << F) - Q_lr(mU)            # lr as fixed-point
    w'  = sr_quant(thU >> F)                 # back on the frozen grid
    eU  = thU - (w' << F)                    # quantization error, fine units
    mU' = mU + Q_alpha(eU)                   # ECO injection, alpha fixed-point
    (m_i8, s_m) = sr_quant_dynamic(mU')      # momentum back to int8

Biases (int32 acc domain) get plain integer SGD, no momentum.
"""

from __future__ import annotations

import torch

from .fixedpoint import (
    _check_int,
    apply_mult_shift,
    int_amax,
    quantize_multiplier,
    to_int8_sat,
)

__all__ = ["I8ECO"]

FRAC_BITS = 12
BETA_Q = 1 << 16  # beta in 16-bit fixed-point
LR_Q = 1 << 16  # lr in 16-bit fixed-point (requires lr < 2)


class I8ECO:
    """Fully integer ECO-SGDM for ``I8Linear`` layers. Zero fp32 state or temporaries."""

    def __init__(
        self,
        layers,
        lr: float = 0.5,
        momentum: float = 0.9,
        grad_clip: float | None = 1.0,
    ):
        assert momentum > 0, "ECO requires beta > 0"
        assert 0.0 < lr < 2.0, "lr must fit 16-bit fixed-point (0, 2)"
        self.layers = list(layers)
        self.lr = float(lr)
        self.beta = float(momentum)
        self.grad_clip = grad_clip
        self.m_i8 = [torch.zeros_like(layer.w_i8) for layer in self.layers]
        self.s_m = [1.0 for _ in self.layers]
        self.step_count = 0
        self.stats = {"skipped": 0, "mean_abs_e": 0.0}

    def set_lr(self, lr: float):
        assert 0.0 < lr < 2.0, "lr must fit 16-bit fixed-point (0, 2)"
        self.lr = float(lr)

    def _clip_grad_int(self, g_i8: torch.Tensor, s_g: float) -> tuple[torch.Tensor, float]:
        # integer grad clip: compare int amax against clip/s_g in scalar math
        if self.grad_clip is None:
            return g_i8, s_g
        import math

        sq_sum = int((g_i8.to(torch.int64) ** 2).sum().item())
        n_real = math.sqrt(sq_sum) * s_g
        if n_real > self.grad_clip:
            # rescale by shrinking the scalar gain (integer tensor untouched)
            return g_i8, s_g * (self.grad_clip / n_real)
        return g_i8, s_g

    @torch.no_grad()
    def step(self):
        b = self.beta
        alpha = (1.0 / self.lr) * (1.0 - 1.0 / b)
        b_q = round(b * BETA_Q)
        nb_q = round((1.0 - b) * BETA_Q)
        lr_q = round(self.lr * LR_Q)
        a_q = round(alpha * BETA_Q)
        self.step_count += 1
        tot_e, n_e = 0.0, 0
        F = FRAC_BITS
        for i, layer in enumerate(self.layers):
            g_i8 = layer._gw_i8
            if g_i8 is None:
                continue
            _check_int(g_i8, "weight grad")
            s_g = float(layer._gw_s)
            g_i8, s_g = self._clip_grad_int(g_i8, s_g)
            s_m = self.s_m[i]
            w_i8 = layer.w_i8
            sw = layer.sw
            out = w_i8.shape[0]

            m_new_rows = []
            # pass 1: updates + errors (per-row int64); momentum scale derived after
            rows = []
            for o in range(out):
                U_num = sw[o]  # real fine unit = sw[o] / 2**F; keep numerator, shift by F
                # m, g into fine units U via fixed-point ratios
                mU = apply_mult_shift(
                    self.m_i8[i][o].to(torch.int64),
                    *quantize_multiplier(s_m * (1 << F) / U_num),
                    stochastic=False,
                )
                gU = apply_mult_shift(
                    g_i8[o].to(torch.int64),
                    *quantize_multiplier(s_g * (1 << F) / U_num),
                    stochastic=False,
                )
                m_t = (mU * b_q + gU * nb_q) >> 16  # int64, beta in 16-bit fp
                thU = (w_i8[o].to(torch.int64) << F) - ((m_t * lr_q) >> 16)
                # stochastic rounding onto the frozen grid (grid step = 2**F in U)
                nudge = torch.randint(0, 1 << F, thU.shape, dtype=torch.int64)
                w_new = torch.div(thU + nudge, 1 << F, rounding_mode="floor")
                w_new = torch.clamp(w_new, -127, 127)
                eU = thU - (w_new << F)
                m_new = m_t + ((eU * a_q) >> 16)
                rows.append((w_new.to(torch.int8), m_new, eU))
                tot_e += (int(eU.abs().sum().item()) / eU.numel()) * U_num / (1 << F)
                n_e += 1

            # momentum scale from real amax: |m_new| * U_o
            amax_real = 0.0
            for (w_new, m_new, _), o in zip(rows, range(out)):
                a = int(m_new.abs().max().item()) * sw[o] / (1 << F)
                amax_real = max(amax_real, a)
            s_m_new = amax_real / 127.0 if amax_real > 0 else 1.0
            self.s_m[i] = s_m_new

            # pass 2: commit weights (frozen grid untouched) + momentum to int8
            for (w_new, m_new, _), o in zip(rows, range(out)):
                w_i8[o].copy_(w_new)
                r = (sw[o] / (1 << F)) / s_m_new if s_m_new > 0 else 0.0
                mq, sq = quantize_multiplier(r)
                self.m_i8[i][o].copy_(
                    to_int8_sat(apply_mult_shift(m_new, mq, sq, stochastic=True))
                )

            # bias: integer SGD in the acc domain (no momentum)
            if layer.use_bias:
                assert isinstance(layer.sx, float)
                gb = layer._gb_sum  # int64, gain s_gb
                s_gb = float(layer._gb_s)
                for o in range(out):
                    r_b = self.lr * s_gb / (layer.sx * sw[o])
                    mq, sq = quantize_multiplier(r_b)
                    delta = apply_mult_shift(
                        gb[o].unsqueeze(0), mq, sq, stochastic=True
                    ).to(torch.int32)
                    layer.b_acc[o] -= delta[0]

        if n_e:
            self.stats["mean_abs_e"] = tot_e / n_e

    @torch.no_grad()
    def zero_grad(self):
        for layer in self.layers:
            layer._gw_i8 = None

    def memory_bytes(self) -> int:
        """Stored integer state in bytes (int8 weights live on layers; int8 momentum here)."""
        return sum(m.numel() for m in self.m_i8)
