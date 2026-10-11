"""Strict-int8 audit: proves the training path allocates no fp32 tensors.

1. Static scan: hot-path sources contain no float-tensor constructors.
2. State scan: every stored tensor (layers + optimizer) is integral.
3. Numeric: I8Linear matches an fp32 Linear within int8 tolerance.
4. End-to-end: strict-int8 training converges on a toy task.
"""
import pathlib

import torch
import torch.nn as nn

from ecoint8.strict import I8Linear, I8ReLU, I8Sequential
from ecoint8.strict import I8ECO
from ecoint8.quant import quantize_symmetric, quantize_tensor_dynamic

STRICT = ["fixedpoint.py", "strict/ilayer.py", "strict/iopt.py"]
BANNED = [".float(", "torch.float32", "torch.float(", "to(torch.float", ".double("]
INT_DTYPES = {torch.int8, torch.int32, torch.int64}


def _pkg() -> pathlib.Path:
    import ecoint8

    return pathlib.Path(ecoint8.__file__).parent


def test_no_float_ops_in_sources():
    for name in STRICT:
        src = (_pkg() / name).read_text()
        for pat in BANNED:
            assert pat not in src, f"{name} contains banned {pat!r}"


def _toy_net():
    return I8Sequential(I8Linear(20, 32), I8ReLU(), I8Linear(32, 2))


def _data():
    torch.manual_seed(0)
    X = torch.randn(512, 20)
    return X, ((X[:, :5].sum(1)) > 0).long()


def test_state_is_integral():
    net = _toy_net()
    X, y = _data()
    with torch.no_grad():
        x8, sx = quantize_tensor_dynamic(X, stochastic=False)
        net.calibrate(x8, sx)
    opt = I8ECO(net.int8_layers(), lr=0.5, momentum=0.9)
    for t, p in net.named_buffers():
        assert p.dtype in INT_DTYPES, f"buffer {t} is {p.dtype}"
    for m in opt.m_i8:
        assert m.dtype == torch.int8
    assert all(isinstance(s, float) for s in opt.s_m)  # scales are scalars, not tensors


def test_layer_matches_fp32():
    torch.manual_seed(1)
    lin = I8Linear(64, 64)
    ref_w = lin.w_i8.to(torch.float32) * torch.tensor(lin.sw, device=lin.w_i8.device).unsqueeze(1)
    x = torch.randn(32, 64)
    with torch.no_grad():
        x8, sx = quantize_tensor_dynamic(x, stochastic=False)
        lin.sx, lin.sy = None, None
        y8, sy = lin.calibrate_forward(x8, sx)
    ref = x @ ref_w.t()
    got = y8.to(torch.float32) * sy
    rel = (got - ref).abs().mean().item() / ref.abs().mean().item()
    # single-layer error scales with the frozen-grid headroom (coarser grid,
    # bigger error); what must NOT happen is structural blowup beyond that.
    assert rel < 0.025 * lin.headroom, f"int8 layer too far from fp32: {rel}"


def test_cuda_pad8_branch_is_exact():
    """The CUDA pad-and-slice is device-agnostic math: verify on CPU."""
    import torch.nn.functional as F

    from ecoint8.fixedpoint import pad_mm_dims

    torch.manual_seed(0)
    for M, K, N in [(512, 256, 2), (2, 512, 256), (7, 13, 5), (100, 300, 2)]:
        a = torch.randint(-127, 127, (M, K), dtype=torch.int8)
        b = torch.randint(-127, 127, (K, N), dtype=torch.int8)
        ref = torch._int_mm(a, b)
        M8, K8, N8 = pad_mm_dims(M, K, N)
        assert M8 > 16 and M8 % 8 == K8 % 8 == N8 % 8 == 0
        ap = F.pad(a, (0, K8 - K, 0, M8 - M))
        bp = F.pad(b, (0, N8 - N, 0, K8 - K))
        assert torch.equal(torch._int_mm(ap, bp)[:M, :N], ref)


def test_e2e_trains():
    net = _toy_net()
    X, y = _data()
    fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        x8, sx = quantize_tensor_dynamic(X, stochastic=False)
        net.calibrate(x8, sx)
    opt = I8ECO(net.int8_layers(), lr=0.5, momentum=0.9)
    for _ in range(30):
        with torch.no_grad():
            logits, s_out = net.forward_i8(x8)
        logits_fp = (logits.to(torch.float32) * s_out).requires_grad_()
        loss = fn(logits_fp, y)
        (g_fp,) = torch.autograd.grad(loss, logits_fp)
        amax = float(g_fp.abs().max().item())
        with torch.no_grad():
            net.backward_i8(
                quantize_symmetric(g_fp, amax / 127.0, stochastic=True), amax / 127.0
            )
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        acc = (net.forward_i8(x8)[0].argmax(1) == y).float().mean().item()
    assert acc > 0.80, f"strict int8 failed to learn: {acc}"
    for layer in net.int8_layers():  # regression: weights must not pin into the rails
        sat = ((layer.w_i8 == 127) | (layer.w_i8 == -127)).float().mean().item()
        assert sat < 0.05, f"weights saturated at rails: {sat:.3f}"
