"""Step-time + VRAM benchmark: eager PyTorch fallback vs fused Triton.

Usage (on a CUDA box):
    pip install eco-int8-opt[gpu]
    python3 examples/benchmark_triton.py [--steps 50] [--hidden 1024] [--sweep]

Warmup is deliberately generous (first Triton compile on a cold cache can
take tens of seconds) + a hard synchronize, so timed steps never include
compile time. On a CPU-only box it times the torch fallback and skips Triton.
"""
import argparse
import time

import torch
import torch.nn as nn

from ecoint8 import EcoInt8SGDM, Int8Linear
from ecoint8.triton_kernels import TRITON_AVAILABLE

WARMUP = 15


class Int8Net(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.fc1 = Int8Linear(d, h)
        self.fc2 = Int8Linear(h, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def timed_step(net, opt, X, y, fn, steps: int, use_cuda: bool) -> float:
    for _ in range(WARMUP):  # warmup incl. triton compile, then fence
        out = net(X)
        loss = fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        out = net(X)
        loss = fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if use_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1e3


def vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def run_config(d: int, h: int, n: int, steps: int, dev: str, fn) -> dict:
    torch.manual_seed(0)
    X = torch.randn(n, d, device=dev)
    y = ((X[:, :5].sum(1)) > 0).long()
    row = {}
    backends = ["torch", "auto"] if dev == "cuda" and TRITON_AVAILABLE else ["torch"]
    for backend in backends:
        torch.manual_seed(0)
        net = Int8Net(d, h).to(dev)
        opt = EcoInt8SGDM([net.fc1, net.fc2], lr=0.5, momentum=0.9, backend=backend)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ms = timed_step(net, opt, X, y, fn, steps, dev == "cuda")
        row[opt.last_step_backend] = (ms, vram_mb(), opt.stats["fused_steps"])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--sweep", action="store_true", help="hidden/batch scaling table")
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    print(f"device={dev} triton_available={TRITON_AVAILABLE} warmup={WARMUP}")
    fn = nn.CrossEntropyLoss()
    configs = [(256, 512, 256), (256, 1024, 512), (256, 2048, 1024), (256, 4096, 2048)] \
        if args.sweep else [(256, args.hidden, args.batch)]
    print(f"{'hidden':>7} {'batch':>6} | {'torch ms':>9} {'triton ms':>10} {'speedup':>8} | vram_t/vram_e MB")
    for d, h, n in configs:
        row = run_config(d, h, n, args.steps, dev, fn)
        t_ms = row["torch"][0]
        if "triton" in row:
            f_ms = row["triton"][0]
            print(f"{h:>7} {n:>6} | {t_ms:>9.3f} {f_ms:>10.3f} {t_ms / f_ms:>7.2f}x | "
                  f"{row['triton'][1]:.1f}/{row['torch'][1]:.1f}")
        else:
            print(f"{h:>7} {n:>6} | {t_ms:>9.3f} {'-':>10} {'-':>8} | -/{row['torch'][1]:.1f}")
            print("NOTE: run me on a CUDA box with eco-int8-opt[gpu] for the Triton row.")


if __name__ == "__main__":
    main()
