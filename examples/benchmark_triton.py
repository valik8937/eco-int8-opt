"""Step-time + VRAM benchmark: eager PyTorch fallback vs fused Triton.

Usage (on a CUDA box):
    pip install eco-int8-opt[gpu]
    python3 examples/benchmark_triton.py [--steps 50] [--hidden 1024]

On a CPU-only box it times the torch fallback and skips the Triton rows.
"""
import argparse
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from ecoint8 import EcoInt8SGDM, Int8Linear
from ecoint8.triton_kernels import TRITON_AVAILABLE


class Int8Net(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.fc1 = Int8Linear(d, h)
        self.fc2 = Int8Linear(h, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def timed_step(net, opt, X, y, fn, steps: int, use_cuda: bool) -> float:
    for _ in range(5):  # warmup
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    print(f"device={dev} triton_available={TRITON_AVAILABLE}")
    torch.manual_seed(0)
    D, H, N = 256, args.hidden, args.batch
    X = torch.randn(N, D, device=dev)
    y = ((X[:, :5].sum(1)) > 0).long()
    fn = nn.CrossEntropyLoss()

    for backend in (["torch", "auto"] if cuda and TRITON_AVAILABLE else ["torch"]):
        torch.manual_seed(0)
        net = Int8Net(D, H).to(dev)
        opt = EcoInt8SGDM([net.fc1, net.fc2], lr=0.5, momentum=0.9, backend=backend)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        ms = timed_step(net, opt, X, y, fn, args.steps, cuda)
        print(
            f"backend={opt.last_step_backend:<7} step={ms:>8.3f}ms "
            f"peak_vram={vram_mb():>8.1f}MB fused_steps={opt.stats['fused_steps']}"
        )
    if not (cuda and TRITON_AVAILABLE):
        print("NOTE: run me on a CUDA box with eco-int8-opt[gpu] for the Triton row.")


if __name__ == "__main__":
    main()
