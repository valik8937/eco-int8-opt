"""End-to-end STRICT int8 training. The only fp32 in this file is documented below.

fp32 boundaries (outside the int8 region, unavoidable):
  [in]  input batch quantized once (entry) — user data arrives fp32;
  [out] final logits dequantized (tiny [N,C]) for CrossEntropy + its grad,
        which is immediately quantized back to int8.
Everything between those two lines — forward, backward, optimizer — is
int8/int32/int64 only. Accuracy is even measured on int8 (argmax is
scale-invariant).
"""
import math

import torch
import torch.nn as nn

from ecoint8.strict import I8Linear, I8ReLU, I8Sequential, I8ECO
from ecoint8.fixedpoint import _check_int
from ecoint8.quant import quantize_tensor_dynamic, quantize_symmetric

torch.manual_seed(0)
N, D, H = 2048, 20, 32
X = torch.randn(N, D)
y = ((X[:, :5].sum(1)) > 0).long()
loss_fn = nn.CrossEntropyLoss()
EPOCHS, LR0, BETA = 150, 0.05, 0.9


def cosine_lr(t: int) -> float:
    return LR0 * 0.5 * (1 + math.cos(math.pi * t / EPOCHS)) + 1e-4


def audit(msg: str, *tensors: torch.Tensor):
    for t in tensors:
        _check_int(t, f"{msg}")


net = I8Sequential(I8Linear(D, H), I8ReLU(), I8Linear(H, 2))

# [in] boundary: entry quant of the input batch (fp32 -> int8, once)
with torch.no_grad():
    x0_i8, sx0 = quantize_tensor_dynamic(X, stochastic=False)
    audit("entry", x0_i8)
    net.calibrate(x0_i8, sx0)

opt = I8ECO(net.int8_layers(), lr=LR0 / (1 - BETA), momentum=BETA)

for ep in range(EPOCHS):
    opt.set_lr(cosine_lr(ep) / (1 - BETA))

    # strict integer forward
    with torch.no_grad():
        logits_i8, s_out = net.forward_i8(x0_i8)
        audit("logits", logits_i8)

    # [out] boundary: dequantize tiny logits for CE; grad back to int8 at once
    logits_fp = (logits_i8.to(torch.float32) * s_out).requires_grad_()
    loss = loss_fn(logits_fp, y)
    (g_fp,) = torch.autograd.grad(loss, logits_fp)
    g_i8 = quantize_symmetric(g_fp, float(g_fp.abs().max().item()) / 127.0,
                              stochastic=True)
    s_g = float(g_fp.abs().max().item()) / 127.0
    audit("entry-grad", g_i8)

    with torch.no_grad():
        net.backward_i8(g_i8, s_g)
    opt.step()
    opt.zero_grad()

    if (ep + 1) % 25 == 0 or ep == 0:
        with torch.no_grad():
            pred = net.forward_i8(x0_i8)[0].argmax(1)  # int8 argmax, no fp32
            acc = (pred == y).float().mean().item()
        print(f"ep={ep+1:>4} lr={cosine_lr(ep):.5f} loss={loss.item():.4f} acc={acc:.3f} "
              f"mean|e|={opt.stats['mean_abs_e']:.2e}")

with torch.no_grad():
    acc = (net.forward_i8(x0_i8)[0].argmax(1) == y).float().mean().item()
print(f"\nSTRICT-INT8 final acc={acc:.3f} state={opt.memory_bytes()}B momentum-int8")
print("buffers:", [(n, str(p.dtype)) for n, p in net.named_buffers()])
assert acc > 0.80, "strict int8 died!"
print("STRICT INT8 SURVIVED")
