"""Базове симетричне int8 квантування. Все просто і швидко."""
from __future__ import annotations
import torch

INT8_MAX = 127.0
EPS = 1e-8


def calc_scale(amax: float, clip: float | None = None) -> float:
    """scale = amax / 127. amax=0 -> scale=1 щоб не ділити на нуль (і не здохнути)."""
    a = float(amax)
    if clip is not None:
        a = min(a, clip)
    if a < EPS:
        return 1.0
    return a / INT8_MAX


def quantize_symmetric(x: torch.Tensor, scale: float | torch.Tensor, stochastic: bool = False) -> torch.Tensor:
    """fp32 -> int8. stochastic=True = стохастичне округлення (рятує малі градієнти від занулення)."""
    s = scale + EPS if isinstance(scale, float) else scale + EPS
    v = x / s
    if stochastic:
        # floor(v + u), u~U(0,1): округлення вгору з імовірністю = дробовій частині
        q = torch.floor(v + torch.rand_like(v))
    else:
        q = torch.round(v)
    return torch.clamp(q, -127, 127).to(torch.int8)


def dequantize(q: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


def quantize_tensor_dynamic(x: torch.Tensor, stochastic: bool = False) -> tuple[torch.Tensor, float]:
    """per-tensor квантування з динамічним скейлом по amax. Повертає (int8, scale)."""
    amax = x.detach().abs().max().item()
    return quantize_symmetric(x, calc_scale(amax), stochastic), calc_scale(amax)


def quantize_per_channel(x: torch.Tensor, axis: int = 0, stochastic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """per-channel квантування (для ваг Linear: кожен вихідний нейрон свій scale)."""
    amax = x.detach().abs().amax(dim=tuple(i for i in range(x.dim()) if i != axis), keepdim=True)
    s = torch.clamp(amax / INT8_MAX, min=EPS)
    return quantize_symmetric(x, s, stochastic), s


def saturation_ratio(q: torch.Tensor) -> float:
    """Частка насичених значень (±127). Якщо >10% — діапазон замалий."""
    return ((q == 127) | (q == -127)).float().mean().item()


def is_dead_grad(q: torch.Tensor) -> bool:
    """Градієнт 'здох' — все нулі після квантування."""
    return bool((q == 0).all())
