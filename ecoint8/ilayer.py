"""Strict integer-only linear layers. No fp32 tensors anywhere in this file.

Static-quantization design:

- A ``calibrate`` pass runs once, freezing every scale: each layer adopts its
  input scale (``sx``) from the previous layer's output scale, computes its
  output scale (``sy``) once, and quantizes its bias into the int32
  accumulator domain against ``sx * s_w``. After that the grids never move —
  which is exactly what ECO theory requires.
- Scales are plain Python floats (``self.sx``, ``self.sy``, ``self.sw`` list),
  deliberately NOT tensors: zero fp32 allocations, even for bookkeeping.
  (Consequence: persist modules with ``torch.save(model)``, not state_dict.)
- No autograd: int tensors cannot carry grads. Forward caches its input,
  ``backward_i8`` is called explicitly in reverse (see ``I8Sequential``).
- Every kernel input/output is dtype-asserted integral — that is the guarantee.

Documented fp32 boundaries (outside the strict region): quantizing the input
batch once, dequantizing final logits for the loss, and one-time float math
when converting a pretrained fp32 layer (``i8_from_float`` in examples).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .fixedpoint import (
    _check_int,
    apply_mult_shift,
    dynamic_scale_from_int,
    int_amax,
    pad_mm_dims,
    quantize_multiplier,
    to_int8_sat,
)

__all__ = ["I8Linear", "I8ReLU", "I8Sequential"]


def _int_mm(a8: torch.Tensor, b8: torch.Tensor) -> torch.Tensor:
    """int8 x int8 -> int32. On CUDA pads dims (multiple of 8, M > 16); exact op."""
    _check_int(a8, "mm lhs")
    _check_int(b8, "mm rhs")
    a8 = a8.contiguous()
    b8 = b8.contiguous()
    if a8.is_cuda:
        M, K = a8.shape
        N = b8.shape[1]
        M8, K8, N8 = pad_mm_dims(M, K, N)
        if (M8, K8, N8) != (M, K, N):
            a8 = torch.nn.functional.pad(a8, (0, K8 - K, 0, M8 - M))
            b8 = torch.nn.functional.pad(b8, (0, N8 - N, 0, K8 - K))
            return torch._int_mm(a8, b8)[:M, :N]
    return torch._int_mm(a8, b8)


def _common_shift_requant(
    acc: torch.Tensor, ratios: list[float], stochastic: bool
) -> torch.Tensor:
    """Requantize int32 ``acc`` [N, out] with per-channel real ratios -> int8.

    One shared shift (minimum over channels), per-channel int32 multipliers
    shifted down to match. Integer tensor math only.
    """
    _check_int(acc, "acc")
    ms = [quantize_multiplier(r) for r in ratios]
    com = min(s for _, s in ms)
    mults = torch.tensor([m >> (s - com) for m, s in ms], dtype=torch.int32)
    y64 = apply_mult_shift(acc.to(torch.int64), mults, com, stochastic)
    return to_int8_sat(y64)


def i8_relu(x_i8: torch.Tensor) -> torch.Tensor:
    """ReLU on int8: integer clamp. Scale unchanged."""
    _check_int(x_i8, "relu input")
    return torch.clamp(x_i8, min=0)


class I8Linear(nn.Module):
    """Integer-only linear layer: int8 in -> int8 out, int32 accumulate."""

    def __init__(self, in_f: int, out_f: int, bias: bool = True):
        super().__init__()
        w_fp = torch.randn(out_f, in_f) * (2.0 / in_f) ** 0.5
        amax = w_fp.abs().amax(dim=1)
        sw = (amax / 127.0).clamp_min(1e-8)
        w_i8 = torch.clamp(torch.round(w_fp / sw.unsqueeze(1)), -127, 127).to(torch.int8)
        self.register_buffer("w_i8", w_i8)
        self.sw: list[float] = [float(v) for v in sw]  # plain floats, NOT a tensor
        self.register_buffer("b_acc", torch.zeros(out_f, dtype=torch.int32))
        self.use_bias = bias
        self.sx: float | None = None  # frozen input scale (set by calibrate)
        self.sy: float | None = None  # frozen output scale (set by calibrate)
        self._cache_x: torch.Tensor | None = None

    @property
    def out_features(self) -> int:
        return self.w_i8.shape[0]

    # -- calibration (once) -------------------------------------------------
    def calibrate_forward(self, x_i8: torch.Tensor, s_x: float) -> tuple[torch.Tensor, float]:
        """First-and-only dynamic pass: freeze ``sx``/``sy``, then behave static."""
        _check_int(x_i8, "calibration input")
        self.sx = float(s_x)
        acc = _int_mm(x_i8, self.w_i8.t())
        if self.use_bias:
            acc = acc + self.b_acc  # zeros at this point; real bias set via set_bias_acc
        gain = self.sx * max(self.sw)
        self.sy = dynamic_scale_from_int(int_amax(acc), gain)
        ratios = [(self.sx * sw) / self.sy for sw in self.sw]
        return _common_shift_requant(acc, ratios, stochastic=True), self.sy

    def set_bias_acc(self, b_acc: torch.Tensor) -> None:
        """Install a bias already quantized to the acc domain (int32)."""
        _check_int(b_acc, "bias acc")
        self.b_acc.copy_(b_acc)

    # -- static forward / explicit backward ---------------------------------
    def forward_i8(self, x_i8: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Static integer forward. Returns ``(y_i8, frozen_sy)``."""
        assert self.sx is not None and self.sy is not None, "calibrate first"
        _check_int(x_i8, "layer input")
        acc = _int_mm(x_i8, self.w_i8.t())
        if self.use_bias:
            acc = acc + self.b_acc
        self._cache_x = x_i8
        assert isinstance(self.sx, float) and isinstance(self.sy, float)
        ratios = [(self.sx * sw) / self.sy for sw in self.sw]
        return _common_shift_requant(acc, ratios, stochastic=True), self.sy

    def backward_i8(self, g_i8: torch.Tensor, s_g: float) -> tuple[torch.Tensor, float]:
        """Integer backward. Returns ``(dx_i8, s_dx)``; stashes int8 ``(gw, gb)`` grads."""
        _check_int(g_i8, "grad output")
        assert self._cache_x is not None, "call forward_i8 before backward_i8"
        x_i8 = self._cache_x
        assert isinstance(self.sx, float)
        s_x = self.sx

        # dW: exact scales (both per-tensor) -> int8 grad for the optimizer
        dw_acc = _int_mm(g_i8.t(), x_i8)  # int32, real gain s_g * s_x
        s_gw = dynamic_scale_from_int(int_amax(dw_acc), s_g * s_x)
        m_gw, sh_gw = quantize_multiplier((s_g * s_x) / s_gw)
        self._gw_i8 = to_int8_sat(apply_mult_shift(dw_acc.to(torch.int64), m_gw, sh_gw, True))
        self._gw_s = s_gw

        # dX: per-channel s_w approximated by its mean
        dx_acc = _int_mm(g_i8, self.w_i8)  # int32
        sw_mean = sum(self.sw) / len(self.sw)
        s_dx = dynamic_scale_from_int(int_amax(dx_acc), s_g * sw_mean)
        m_dx, sh_dx = quantize_multiplier((s_g * sw_mean) / s_dx)
        dx_i8 = to_int8_sat(apply_mult_shift(dx_acc.to(torch.int64), m_dx, sh_dx, True))

        # dBias: integer column sums of g_i8 (real gain s_g), stays int64
        self._gb_sum = g_i8.to(torch.int64).sum(dim=0)
        self._gb_s = float(s_g)
        return dx_i8, s_dx


class I8ReLU(nn.Module):
    """ReLU block for I8Sequential with integer mask caching."""

    def __init__(self):
        super().__init__()
        self.mask_cache: torch.Tensor | None = None

    def apply_forward(self, x_i8: torch.Tensor) -> torch.Tensor:
        self.mask_cache = (x_i8 > 0).to(torch.int8)
        return i8_relu(x_i8)

    def backward_mask(self, g_i8: torch.Tensor) -> torch.Tensor:
        _check_int(g_i8, "relu grad")
        assert self.mask_cache is not None
        gm = g_i8.to(torch.int32) * self.mask_cache.to(torch.int32)
        return torch.clamp(gm, -127, 127).to(torch.int8)


class I8Sequential(nn.Module):
    """Manual int8 pipeline: threads int8 tensors + scalar scales, explicit backward."""

    def __init__(self, *blocks: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def int8_layers(self) -> list[I8Linear]:
        return [b for b in self.blocks if isinstance(b, I8Linear)]

    def calibrate(self, x_i8: torch.Tensor, s_x: float) -> tuple[torch.Tensor, float]:
        """One dynamic pass that freezes every layer's scales. Returns (logits, s)."""
        _check_int(x_i8, "calibration input")
        s = float(s_x)
        for blk in self.blocks:
            if isinstance(blk, I8Linear):
                x_i8, s = blk.calibrate_forward(x_i8, s)
            elif isinstance(blk, I8ReLU):
                x_i8 = blk.apply_forward(x_i8)
        return x_i8, s

    def forward_i8(self, x_i8: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Static integer forward over frozen grids. Returns (logits_i8, s_out)."""
        _check_int(x_i8, "net input")
        s: float | None = None
        for blk in self.blocks:
            if isinstance(blk, I8Linear):
                x_i8, s = blk.forward_i8(x_i8)
            elif isinstance(blk, I8ReLU):
                x_i8 = blk.apply_forward(x_i8)
        assert s is not None
        return x_i8, s

    def backward_i8(self, g_i8: torch.Tensor, s_g: float) -> None:
        """Explicit reverse pass; grads land on ``layer._gw_i8/_gb_sum``."""
        _check_int(g_i8, "net grad")
        for blk in reversed(self.blocks):
            if isinstance(blk, I8Linear):
                g_i8, s_g = blk.backward_i8(g_i8, s_g)
            elif isinstance(blk, I8ReLU):
                g_i8 = blk.backward_mask(g_i8)
