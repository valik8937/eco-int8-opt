"""CUDA profiler for the triton backend: which kernel eats the step time?

Usage (CUDA box):  python3 examples/profile_triton.py
Needs nothing beyond torch (uses built-in Kineto profiler).
Paste the printed table back.
"""
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from ecoint8 import EcoInt8SGDM, Int8Linear, to_int8


def build():
    torch.manual_seed(0)
    net = to_int8(nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 2))).cuda()
    opt = EcoInt8SGDM(
        [m for m in net.modules() if isinstance(m, Int8Linear)],
        lr=0.5, momentum=0.9, backend="triton",
    )
    return net, opt


torch.manual_seed(0)
X = torch.randn(512, 256, device="cuda")
y = ((X[:, :5].sum(1)) > 0).long()
fn = nn.CrossEntropyLoss()
net, opt = build()
print("backend check:", opt.last_step_backend or "(runs on first step)")

for _ in range(8):  # warmup incl. triton compile
    out = net(X)
    loss = fn(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
print("warmed up, backend:", opt.last_step_backend)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        out = net(X)
        loss = fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
