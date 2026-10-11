"""VRAM-per-phase probe for strict int8 ECO training.

Canonical usage (CUDA box, e.g. Tesla T4):
    python3 examples/mem_probe.py

Self-contained: pure stdlib + torch, no torchvision, no downloads.
On CPU-only boxes it still runs (slower) untimed as a correctness smoke test.
"""

import os
import sys

# Standalone path bootstrap: allow `python3 examples/mem_probe.py` from the
# repo root without requiring `pip install -e .`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from ecoint8 import I8Linear, I8ReLU, I8Sequential, I8ECO
from ecoint8.quant import quantize_tensor_dynamic

HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")
MiB = 1024.0 * 1024.0


def sync():
    if HAS_CUDA:
        torch.cuda.synchronize()


def reset_peak():
    if HAS_CUDA:
        torch.cuda.reset_peak_memory_stats()


def peak_mb():
    if HAS_CUDA:
        sync()
        return torch.cuda.max_memory_allocated() / MiB
    return float("nan")


def main():
    torch.manual_seed(0)
    print(f"device={DEVICE} has_cuda={HAS_CUDA} torch={torch.__version__}")
    if not HAS_CUDA:
        print("CPU fallback note: no CUDA found; running phases untimed "
              "for correctness smoke only (no VRAM peaks).")

    # 1. Exact benchmark net, calibrated on one synthetic batch.
    net = I8Sequential(
        I8Linear(784, 256),
        I8ReLU(),
        I8Linear(256, 128),
        I8ReLU(),
        I8Linear(128, 10),
    ).to(str(DEVICE))

    # 2. ONE fixed synthetic batch [256, 784] fp32 (+ fixed labels) for everything.
    X_fp = torch.randn(256, 784, device=DEVICE)
    y = torch.randint(0, 10, (256,), device=DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        x0_i8, sx0 = quantize_tensor_dynamic(X_fp, stochastic=False)
        net.calibrate(x0_i8, sx0)
    # Steady-state entry on the FROZEN grid; reuse this one tensor everywhere.
    with torch.no_grad():
        x_i8, _sx = net.quantize_input(X_fp)

    opt = I8ECO(net.int8_layers(), lr=0.08, momentum=0.9)

    def full_step():
        # forward_i8 -> fp32 CE-style boundary grad -> quantize -> backward -> step
        with torch.no_grad():
            logits_i8, s_out = net.forward_i8(x_i8)
        logits_fp = (logits_i8.to(torch.float32) * s_out).requires_grad_()
        loss = loss_fn(logits_fp, y)
        (g_fp,) = torch.autograd.grad(loss, logits_fp)
        g_i8, s_g = quantize_tensor_dynamic(g_fp, stochastic=True)
        with torch.no_grad():
            net.backward_i8(g_i8, s_g)
        opt.step()
        opt.zero_grad()

    # Idle baseline peak: right after calibrate (+opt alloc), before warmup.
    if HAS_CUDA:
        sync()
        idle_peak = torch.cuda.max_memory_allocated() / MiB
        print(f"idle baseline peak (post-calibrate, pre-warmup): {idle_peak:.2f} MB")
    else:
        print("idle baseline peak: n/a (CPU, no CUDA peaks)")

    # 3. Warm up 5 full steps, then reset peak stats.
    for _ in range(5):
        full_step()
    sync()
    reset_peak()

    # 4. Per-phase peaks. sync() before every reading.
    results = {}

    # forward_i8 only
    reset_peak()
    with torch.no_grad():
        logits_i8, s_out = net.forward_i8(x_i8)
    results["forward_i8"] = peak_mb()

    # backward_i8 only: needs a forward first (uncounted -> reset right before).
    with torch.no_grad():
        logits_i8, s_out = net.forward_i8(x_i8)
    logits_fp = (logits_i8.to(torch.float32) * s_out).requires_grad_()
    loss = loss_fn(logits_fp, y)
    (g_fp,) = torch.autograd.grad(loss, logits_fp)
    g_i8, s_g = quantize_tensor_dynamic(g_fp, stochastic=True)
    reset_peak()
    with torch.no_grad():
        net.backward_i8(g_i8, s_g)
    results["backward_i8"] = peak_mb()
    opt.zero_grad()

    # I8ECO.step only: populate grads first (uncounted), reset, then step.
    with torch.no_grad():
        logits_i8, s_out = net.forward_i8(x_i8)
    logits_fp = (logits_i8.to(torch.float32) * s_out).requires_grad_()
    loss = loss_fn(logits_fp, y)
    (g_fp,) = torch.autograd.grad(loss, logits_fp)
    g_i8, s_g = quantize_tensor_dynamic(g_fp, stochastic=True)
    with torch.no_grad():
        net.backward_i8(g_i8, s_g)
    reset_peak()
    opt.step()
    results["opt_step"] = peak_mb()
    opt.zero_grad()

    # one full step
    reset_peak()
    full_step()
    results["full_step"] = peak_mb()

    # 5a. phase -> peak MB table
    print("\nphase -> peak MB:")
    for k in ("forward_i8", "backward_i8", "opt_step", "full_step"):
        v = results[k]
        if HAS_CUDA:
            print(f"  {k:<12} {v:8.2f} MB")
        else:
            print(f"  {k:<12} n/a (CPU, ran untimed for smoke)")

    # 5b. stored-state bytes: optimizer int8 momentum + buffers
    mom_bytes = opt.memory_bytes()
    buf_rows = []
    buf_total = 0
    for name, p in net.named_buffers():
        b = p.numel() * p.element_size()
        buf_total += b
        buf_rows.append((name, str(p.dtype), tuple(p.shape), b))
    print(f"\nstored state: opt.memory_bytes()={mom_bytes} B (int8 momentum)")
    print(f"stored state: buffers total={buf_total} B over {len(buf_rows)} buffers")
    for name, dt, sh, b in sorted(buf_rows, key=lambda r: -r[3]):
        print(f"  {name:<28} {dt:<12} {str(sh):<22} {b:>10} B")

    # 5c. memory_summary truncated to top ~15 lines (names top offenders).
    if HAS_CUDA:
        sync()
        summary = torch.cuda.memory_summary()
        lines = summary.splitlines()
        print("\nmemory_summary (top ~15 lines):")
        print("\n".join(lines[:15]))
    else:
        print("\nmemory_summary: n/a (CPU, no CUDA)")


if __name__ == "__main__":
    main()
