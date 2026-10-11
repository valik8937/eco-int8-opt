"""Integer-only fixed-point helpers. No fp32 tensors are created anywhere here.

Convention (TFLite-style): a real-valued scale ratio ``r >= 0`` is decomposed
into ``(mult, shift)`` with ``mult`` in ``[2**30, 2**31)`` such that
``r ~= mult / 2**shift``. All tensor math below is int8/int32/int64 only.
Scale bookkeeping (deriving ``mult``/``shift`` from ratios) uses plain Python
scalars — never tensors.
"""

from __future__ import annotations

import math

import torch

_INT8_LO, _INT8_HI = -127, 127


def _check_int(t: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    assert t.dtype in (torch.int8, torch.int32, torch.int64), (
        f"{name} must be integral, got {t.dtype} — fp32 leaked into the int path"
    )
    return t


def quantize_multiplier(r: float) -> tuple[int, int]:
    """Decompose a non-negative ratio into ``(mult, shift)`` with ``r ~= mult / 2**shift``."""
    assert r >= 0.0, "scale ratios must be non-negative"
    if r == 0.0:
        return 0, 0
    mantissa, exponent = math.frexp(r)  # r = mantissa * 2**exponent, mantissa in [0.5, 1)
    shift = exponent
    mult = round(mantissa * (1 << 31))
    if mult == (1 << 31):  # corner case: mantissa rounds up to 1.0
        mult //= 2
        shift += 1
    return mult, 31 - shift  # apply as: y = (x * mult) >> total_shift


def apply_mult_shift(
    x: torch.Tensor, mult: int | torch.Tensor, shift: int, stochastic: bool = False
) -> torch.Tensor:
    """Integer ``y = round(x * mult / 2**shift)``. ``x`` int32/int64 -> int64.

    ``shift`` may be negative (= left shift). ``mult`` may be a negative Python
    int, or an int32/int64 tensor for per-channel multipliers (one shared shift).
    Stochastic rounding is unbiased: ``floor(v + u)``, ``u ~ U[0,1)`` via randint.
    """
    _check_int(x, "x")
    xi = x.to(torch.int64)
    if isinstance(mult, torch.Tensor):
        _check_int(mult, "mult")
        prod = xi * mult.to(device=xi.device, dtype=torch.int64)
    else:
        prod = xi * mult if mult != 1 else xi
    if shift >= 0:
        denom = 1 << shift
        if stochastic:
            nudge = torch.randint(0, denom, prod.shape, dtype=torch.int64, device=prod.device)
            prod = prod + nudge
        else:
            prod = prod + (denom >> 1)  # round-half-up via floor
        return torch.div(prod, denom, rounding_mode="floor")
    else:
        return prod << (-shift)


def to_int8_sat(x: torch.Tensor) -> torch.Tensor:
    """Clamp an integral tensor into int8 (symmetric, [-127, 127])."""
    _check_int(x, "x")
    return torch.clamp(x, _INT8_LO, _INT8_HI).to(torch.int8)


def int_amax(x: torch.Tensor) -> int:
    """Max abs as a plain Python int (scalar bookkeeping, no fp32)."""
    _check_int(x, "x")
    return int(x.abs().max().item())


def _pad8(n: int) -> int:
    return (n + 7) // 8 * 8


def pad_mm_dims(m: int, k: int, n: int) -> tuple[int, int, int]:
    """Padded dims for CUDA ``torch._int_mm``.

    Constraints (from CUDA errors): every dim a multiple of 8, and the row
    count M strictly greater than 16. Zero-padding is bit-exact: padded rows
    contribute nothing, and the result is sliced back. Device-agnostic math.
    """
    return max(32, _pad8(m)), _pad8(k), _pad8(n)


def dynamic_scale_from_int(acc_amax: int, real_gain: float) -> float:
    """Output scale (Python float) from an integer amax and a real-valued gain.

    ``real_gain`` maps accumulator units to real values, e.g. ``s_x * max(s_w)``.
    Returns ``max_abs_real / 127``. Scalar math only.
    """
    real_amax = acc_amax * real_gain
    if real_amax <= 0.0:
        return 1.0
    return real_amax / 127.0
