"""Бенчмарк: fp32 vs _int_mm (навчальний) vs FBGEMM prepacked (inference) на CPU."""
import time
import torch
from ecoint8.quant import quantize_tensor_dynamic, quantize_per_channel
from ecoint8.deploy import FbgemmInt8Linear

torch.manual_seed(0)
for (M, K, N) in [(512, 512, 512), (1024, 1024, 1024), (2048, 512, 1024)]:
    xf = torch.randn(M, K)
    wf = torch.randn(N, K)
    x8, sx = quantize_tensor_dynamic(xf)
    w8, sw = quantize_per_channel(wf, axis=0)

    for _ in range(5):
        torch._int_mm(x8, w8.t())
        xf @ wf.t()
    fb = FbgemmInt8Linear(w8, sw, None)
    for _ in range(5):
        fb(xf)
    it = 20

    t0 = time.time()
    for _ in range(it):
        xf @ wf.t()
    t_fp = (time.time() - t0) / it * 1e3

    t0 = time.time()
    for _ in range(it):
        torch._int_mm(x8, w8.t()).float() * sx * sw.t()
    t_mm = (time.time() - t0) / it * 1e3

    t0 = time.time()
    for _ in range(it):
        fb(xf)
    t_fb = (time.time() - t0) / it * 1e3

    ref = xf @ wf.t()
    got_mm = (torch._int_mm(x8, w8.t()).float() * sx * sw.t())
    rel_mm = ((got_mm - ref).abs().mean() / ref.abs().mean()).item()
    rel_fb = ((fb(xf) - ref).abs().mean() / ref.abs().mean()).item()
    print(f"[{M}x{K}@{K}x{N}] fp32={t_fp:.2f}ms _int_mm={t_mm:.2f}ms "
          f"fbgemm={t_fb:.2f}ms (x{t_fp / t_fb:.2f}) err_mm={rel_mm:.4f} err_fb={rel_fb:.4f}")
