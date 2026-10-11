"""I8ECO: integer-only ECO step. No fp32 tensors anywhere in this file.

Working precision: per-channel fine units ``U_o = sw_o / 2**F`` (F=12),
int64 accumulators. Every real-valued factor (lr, beta, alpha, scale ratios)
becomes an integer ``(mult, shift)`` pair via ``quantize_multiplier`` —
scalar Python math only. Stochastic rounding via ``torch.randint``.

One step per weight row ``o`` (``U = sw/2**F``), vectorized over rows —
per-row ratios differ only through ``sw[o]``, so each shares one common
shift (``_col_mults``) and all tensor math is whole-layer::

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

from ..fixedpoint import (
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


def _col_mults(ratios: list[float], dev: torch.device) -> tuple[torch.Tensor, int]:
    """Per-row ``(mult, shift)`` pairs sharing one common shift.

    Same math as the scalar ``quantize_multiplier`` path, but the tensor
    math stays whole-layer: one shared shift (minimum over rows), per-row
    int32 multipliers shifted down to match, as an ``[out, 1]`` column that
    broadcasts over ``[out, in]`` in ``apply_mult_shift``. The ratios list
    is scalar Python math (no kernels, no syncs).
    """
    ms = [quantize_multiplier(r) for r in ratios]
    com = min(s for _, s in ms)
    mults = torch.tensor([m >> (s - com) for m, s in ms], dtype=torch.int32, device=dev)
    return mults.unsqueeze(1), com


class I8ECO:
    """Fully integer ECO-SGDM for ``I8Linear`` layers. Zero fp32 state or temporaries."""

    def __init__(
        self,
        layers,
        lr: float = 0.5,
        momentum: float = 0.9,
        grad_clip: float | None = 1.0,
        chunk_rows: int = 64,
    ):
        assert momentum > 0, "ECO requires beta > 0"
        assert 0.0 < lr < 2.0, "lr must fit 16-bit fixed-point (0, 2)"
        assert chunk_rows >= 1, "chunk_rows must be positive"
        self.layers = list(layers)
        self.lr = float(lr)
        self.beta = float(momentum)
        self.grad_clip = grad_clip
        # Row chunk size for the step: whole-layer tensor math needs ~14
        # full-size int64 temporaries at once (~21 MB for a 256x784 layer);
        # chunking bounds that peak (64 rows -> ~5 MB) at the cost of a few
        # more kernel launches. Same math, same distribution, only the
        # stochastic-rounding stream differs from unchunked.
        self.chunk_rows = int(chunk_rows)
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
            dev = w_i8.device
            m_i8 = self.m_i8[i]

            # Whole-layer tensor math in row chunks (see chunk_rows): per-row
            # ratios differ only through sw[o], so they share one common
            # shift each; slices of the precomputed columns stay valid.
            # Pass 1 commits weights and stashes m_new chunks (one full-size
            # int64 — the only whole-layer transient); pass 2 requants
            # momentum once s_m_new is known from the row amaxes.
            mults_m, com_m = _col_mults([s_m * (1 << F) / sw_o for sw_o in sw], dev)
            mults_g, com_g = _col_mults([s_g * (1 << F) / sw_o for sw_o in sw], dev)
            rowmax_all: list[int] = []
            rowsum_all: list[int] = []
            m_chunks: list[torch.Tensor] = []
            rowlen = g_i8.shape[1]
            for c in range(0, out, self.chunk_rows):
                sl = slice(c, min(c + self.chunk_rows, out))
                mU = apply_mult_shift(
                    m_i8[sl].to(torch.int64), mults_m[sl], com_m, stochastic=False
                )
                gU = apply_mult_shift(
                    g_i8[sl].to(torch.int64), mults_g[sl], com_g, stochastic=False
                )
                m_t = (mU * b_q + gU * nb_q) >> 16  # int64, beta in 16-bit fp
                thU = (w_i8[sl].to(torch.int64) << F) - ((m_t * lr_q) >> 16)
                # stochastic rounding onto the frozen grid (grid step = 2**F in U)
                nudge = torch.randint(
                    0, 1 << F, thU.shape, dtype=torch.int64, device=thU.device
                )
                w_new = torch.clamp(
                    torch.div(thU + nudge, 1 << F, rounding_mode="floor"), -127, 127
                )
                eU = thU - (w_new << F)
                m_new = m_t + ((eU * a_q) >> 16)
                rowmax_all.extend(m_new.abs().amax(dim=1).tolist())
                rowsum_all.extend(eU.abs().sum(dim=1).tolist())
                w_i8[sl].copy_(w_new.to(torch.int8))
                m_chunks.append(m_new)
            # momentum scale from real amax: max_o(max|m_new_o| * U_o)
            amax_real = max(r * sw_o / (1 << F) for r, sw_o in zip(rowmax_all, sw))
            s_m_new = amax_real / 127.0 if amax_real > 0 else 1.0
            self.s_m[i] = s_m_new
            # stats: mean over rows of mean|e| in real units (same as before:
            # per-row means summed, divided by the row count in n_e)
            tot_e += sum(s * sw_o for s, sw_o in zip(rowsum_all, sw)) / rowlen / (1 << F)
            n_e += out
            # pass 2: momentum back into int8 (frozen weight grid untouched)
            mults_mm, com_mm = _col_mults(
                [(sw_o / (1 << F)) / s_m_new if s_m_new > 0 else 0.0 for sw_o in sw],
                dev,
            )
            for m_new, c in zip(m_chunks, range(0, out, self.chunk_rows)):
                sl = slice(c, min(c + self.chunk_rows, out))
                self.m_i8[i][sl].copy_(
                    to_int8_sat(
                        apply_mult_shift(m_new, mults_mm[sl], com_mm, stochastic=True)
                    )
                )

            # bias: integer SGD in the acc domain (no momentum)
            if layer.use_bias:
                assert isinstance(layer.sx, float)
                gb = layer._gb_sum  # int64 [out], gain s_gb
                s_gb = float(layer._gb_s)
                mults_b, com_b = _col_mults(
                    [self.lr * s_gb / (layer.sx * sw_o) for sw_o in sw], dev
                )
                delta = apply_mult_shift(
                    gb.unsqueeze(1), mults_b, com_b, stochastic=True
                ).to(torch.int32)
                layer.b_acc -= delta.squeeze(1)

        if n_e:
            self.stats["mean_abs_e"] = tot_e / n_e

    @torch.no_grad()
    def zero_grad(self):
        for layer in self.layers:
            layer._gw_i8 = None

    def memory_bytes(self) -> int:
        """Stored integer state in bytes (int8 weights live on layers; int8 momentum here)."""
        return sum(m.numel() for m in self.m_i8)
