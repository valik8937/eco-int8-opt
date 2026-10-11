"""Fused Triton kernels for :class:`EcoInt8SGDM`. Optional dependency.

Install with ``pip install eco-int8-opt[gpu]``. CPU-only setups never touch
this module's kernels (import is guarded by ``TRITON_AVAILABLE``).

Design (one ``EcoInt8SGDM`` step, per weight element, everything in registers)::

    g_hat   = SR((g + w*sw*wd) / sg) * sg        # grad quant (wd fused)
    m_tilde = beta * m * sm + (1-beta) * g_hat
    th      = w * sw - lr * m_tilde
    w_new   = SR(th / sw)                        # frozen grid
    e       = th - w_new * sw
    m_new   = m_tilde + alpha * e                # ECO injection

Notes / parity with the torch fallback in ``eco.py``:

- ``grad_clip`` is intentionally NOT folded in: in the CPU path the clip
  coefficient cancels exactly (amax is recomputed after clipping), so the
  clip is a no-op there, and the triton path mirrors that bit-for-bit.
- The momentum scale ``sm`` is a persistent 0-dim GPU scalar per layer,
  updated in-place every step (``sm.copy_(new)``) — never stale, never synced.
- Stochastic rounding draws come from ``tl.rand(seed, offsets)`` —
  deterministic in ``(seed, element, draw-site)``, so both passes below draw
  IDENTICAL values and agree exactly. Two launches per step, zero full-size
  fp32 temporaries, zero host syncs:

1. ``eco_amax_kernel`` — recompute ``m_new`` per tile, block-max ``|m_new|``
   into a small persistent scratch (<=8192 floats, allocated once). Gives the
   fresh momentum scale ``s_m_new = max / 127`` as a 0-dim GPU scalar.
2. ``eco_fused_kernel`` — same math + stores ``w_i8``/``m_i8`` (SR via randint).

Per-channel ``sw`` is indexed as ``sw[off // in_features]`` — no duplication.
Biases are intentionally NOT fused (tiny fp32 torch SGD, already CUDA-native).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # CPU-only install: triton path simply unavailable
    TRITON_AVAILABLE = False

BLOCK = 1024
MAX_SCRATCH = 8192  # covers layers up to 8M params at BLOCK=1024


if TRITON_AVAILABLE:

    @triton.jit
    def _eco_elem(w, m, g, sw, sm, sg, wd, lr, beta, alpha, r_g, r_w):
        """One element of the ECO update. All fp32 values live in registers."""
        gv = (g + w * sw * wd) / sg
        gq = tl.minimum(tl.maximum(tl.floor(gv + r_g), -127.0), 127.0)
        g_hat = gq * sg
        m_tilde = beta * m * sm + (1.0 - beta) * g_hat
        th = w * sw - lr * m_tilde
        wq = tl.minimum(tl.maximum(tl.floor(th / sw + r_w), -127.0), 127.0)
        e = th - wq * sw
        m_new = m_tilde + alpha * e
        return wq.to(tl.int8), m_new

    @triton.jit
    def eco_amax_kernel(
        w_ptr, m_ptr, g_ptr, sw_ptr, sm_ptr, sg_ptr,
        wd, lr, beta, alpha,
        in_features, n_elements, partials_ptr, seed,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        w = tl.load(w_ptr + offs, mask=mask, other=0).to(tl.float32)
        m = tl.load(m_ptr + offs, mask=mask, other=0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        ch = offs // in_features
        sw = tl.load(sw_ptr + ch, mask=mask, other=1.0)
        sm = tl.load(sm_ptr)
        sg = tl.load(sg_ptr)
        base = offs * 2
        wq, m_new = _eco_elem(
            w, m, g, sw, sm, sg, wd, lr, beta, alpha,
            tl.rand(seed, base + 0), tl.rand(seed, base + 1),
        )
        tile_max = tl.max(tl.where(mask, tl.abs(m_new), 0.0), 0)
        tl.store(partials_ptr + pid, tile_max)

    @triton.jit
    def eco_fused_kernel(
        w_ptr, m_ptr, g_ptr, sw_ptr, sm_ptr, sg_ptr, sm_new_ptr,
        wd, lr, beta, alpha,
        in_features, n_elements, seed,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        w = tl.load(w_ptr + offs, mask=mask, other=0).to(tl.float32)
        m = tl.load(m_ptr + offs, mask=mask, other=0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        ch = offs // in_features
        sw = tl.load(sw_ptr + ch, mask=mask, other=1.0)
        sm = tl.load(sm_ptr)
        sg = tl.load(sg_ptr)
        sm_new = tl.load(sm_new_ptr) + 1e-8  # guard against div-by-zero
        base = offs * 3
        wq, m_new = _eco_elem(
            w, m, g, sw, sm, sg, wd, lr, beta, alpha,
            tl.rand(seed, base + 0), tl.rand(seed, base + 1),
        )
        mq = tl.minimum(
            tl.maximum(tl.floor(m_new / sm_new + tl.rand(seed, base + 2)), -127.0), 127.0
        )
        tl.store(w_ptr + offs, wq, mask=mask)
        tl.store(m_ptr + offs, mq.to(tl.int8), mask=mask)


def _require_triton():
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton path needs `pip install eco-int8-opt[gpu]` (triton)")


def triton_eco_step_(
    w_i8: torch.Tensor,
    m_i8: torch.Tensor,
    g_fp32: torch.Tensor,
    sw_flat: torch.Tensor,
    sm_t: torch.Tensor,
    *,
    lr: float,
    beta: float,
    wd: float,
    seed: int,
    scratch: torch.Tensor,
) -> str:
    """Fused in-place ECO step. No host syncs, no full-size fp32 temporaries.

    ``sm_t`` is the persistent 0-dim per-layer momentum scale, updated in
    place (no ``.item()`` anywhere on this path). Always returns ``"ok"``.
    """
    _require_triton()
    assert w_i8.is_cuda and w_i8.dtype == torch.int8 and w_i8.is_contiguous()
    assert m_i8.shape == w_i8.shape and m_i8.dtype == torch.int8
    assert g_fp32.shape == w_i8.shape and g_fp32.dtype == torch.float32
    assert sw_flat.dtype == torch.float32 and sw_flat.numel() == w_i8.shape[0]
    assert sm_t.numel() == 1 and sm_t.dtype == torch.float32
    n = w_i8.numel()
    in_features = w_i8.shape[1]
    grid = (n + BLOCK - 1) // BLOCK
    assert grid <= scratch.numel(), f"layer too large for scratch ({n} elems)"

    sg_t = torch.maximum(g_fp32.abs().max(), 1e-12) / 127.0  # 0-dim, no sync
    alpha = (1.0 / lr) * (1.0 - 1.0 / beta)
    eco_amax_kernel[(grid,)](
        w_i8, m_i8, g_fp32, sw_flat, sm_t, sg_t,
        float(wd), float(lr), float(beta), alpha,
        in_features, n, scratch, int(seed), BLOCK,
    )
    sm_new_t = scratch[:grid].max().unsqueeze(0) / 127.0  # 0-dim GPU scalar
    eco_fused_kernel[(grid,)](
        w_i8, m_i8, g_fp32, sw_flat, sm_t, sg_t, sm_new_t,
        float(wd), float(lr), float(beta), alpha,
        in_features, n, int(seed), BLOCK,
    )
    sm_t.copy_(sm_new_t)
    return "ok"
