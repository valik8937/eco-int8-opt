"""Diagnose torch-vs-triton divergence: SR-noise chaos (H0) or kernel bug (H1)?

Three runs, identical init/data, 12 steps (divergence is visible by step 5):
  A: backend="torch",  SR stream seeded S1
  B: backend="torch",  SR stream seeded S2  (different noise, same backend)
  C: backend="triton", SR via philox

Verdict:
  - B ~= A but C far  -> H1, systematic triton bug (then dump localizes it).
  - B diverges like C -> H0, ECO dynamics at this lr amplify ANY SR noise
    (algorithmic sensitivity, not a kernel bug).

Usage (CUDA box):  python3 examples/debug_parity.py
"""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from ecoint8 import EcoInt8SGDM, Int8Linear, to_int8

STEPS = 12


def make_net():
    torch.manual_seed(0)
    return to_int8(nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))).cuda()


def run(backend: str, rng_seed: int):
    net = make_net()
    layers = [m for m in net.modules() if isinstance(m, Int8Linear)]
    torch.manual_seed(rng_seed)  # SR stream only from here (init already fixed above)
    opt = EcoInt8SGDM(layers, lr=0.03, momentum=0.9, backend=backend)
    hist = []
    for _ in range(STEPS):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(X), y)
        loss.backward()
        opt.step()
        hist.append(loss.item())
    assert opt.last_step_backend == backend, opt.last_step_backend
    snap = {}
    for i, l in enumerate(layers):
        m = opt.m_i8[i]
        sm = opt.s_m[i] if backend == "torch" else float(opt._tri_sm[i].item())
        snap[i] = (l.w_i8.clone(), m.clone(), sm)
    return hist, snap


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).norm().item() / max(a.float().norm().item(), 1e-12)


torch.manual_seed(42)
X = torch.randn(64, 64, device="cuda")
y = torch.randint(0, 10, (64,), device="cuda")

hA, sA = run("torch", 100)
hB, sB = run("torch", 777)
hC, sC = run("triton", 100)

print(f"{'step':>4} | {'A torch(s1)':>11} | {'B torch(s2)':>11} | {'C triton':>11}")
for t in range(STEPS):
    print(f"{t+1:>4} | {hA[t]:>11.4f} | {hB[t]:>11.4f} | {hC[t]:>11.4f}")

gapAB = abs(hA[-1] - hB[-1])
gapAC = abs(hA[-1] - hC[-1])
print(f"\nfinal-loss gaps: |A-B|={gapAB:.4f} (noise) vs |A-C|={gapAC:.4f} (backend)")
for i in (0, 1):
    wa, ma, sma = sA[i]
    wb, mb, smb = sB[i]
    wc, mc, smc = sC[i]
    print(
        f"layer{i}: ||wA-wB||/||w||={rel(wa, wb):.4f}  ||wA-wC||/||w||={rel(wa, wc):.4f}  "
        f"s_m: cpuA={sma:.3e} cpuB={smb:.3e} tri={smc:.3e}"
    )
if gapAC > 3 * max(gapAB, 1e-6):
    print("VERDICT: H1 (systematic triton bug) — backend gap dwarfs noise gap.")
else:
    print("VERDICT: H0 (SR-noise chaos) — different noise alone diverges as much.")
