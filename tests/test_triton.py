"""Triton backend gating + parity. GPU numerics test skips without CUDA."""
import pytest
import torch
import torch.nn as nn

from ecoint8 import EcoInt8SGDM, Int8Linear
from ecoint8.triton_kernels import TRITON_AVAILABLE

HAS_GPU = torch.cuda.is_available() and TRITON_AVAILABLE


def _net():
    class N(nn.Module):
        def __init__(s):
            super().__init__()
            s.fc1 = Int8Linear(20, 32)
            s.fc2 = Int8Linear(32, 2)

        def forward(s, x):
            return s.fc2(torch.relu(s.fc1(x)))

    return N()


def test_triton_module_importable_no_gpu():
    assert isinstance(TRITON_AVAILABLE, bool)


def test_auto_falls_back_to_torch_on_cpu():
    torch.manual_seed(0)
    net = _net()
    opt = EcoInt8SGDM([net.fc1, net.fc2], lr=0.5, momentum=0.9)  # backend="auto"
    X = torch.randn(64, 20)
    y = ((X[:, :5].sum(1)) > 0).long()
    loss = nn.CrossEntropyLoss()(net(X), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert opt.last_step_backend == "torch"
    assert opt.stats["fused_steps"] == 0


def test_backend_triton_raises_without_cuda_or_triton():
    if HAS_GPU:
        pytest.skip("only meaningful without a triton-capable GPU")
    net = _net()
    opt = EcoInt8SGDM([net.fc1, net.fc2], backend="triton")
    X = torch.randn(64, 20)
    y = ((X[:, :5].sum(1)) > 0).long()
    loss = nn.CrossEntropyLoss()(net(X), y)
    opt.zero_grad()
    loss.backward()
    with pytest.raises(RuntimeError):
        opt.step()


@pytest.mark.skipif(not HAS_GPU, reason="needs CUDA + triton")
def test_triton_matches_torch_parity():
    """Same init/data, 20 steps each backend: SR streams differ (torch.rand vs
    philox), so compare outcomes, not bits. RUN ON GPU BOX."""
    torch.manual_seed(0)
    X = torch.randn(512, 20).cuda()
    y = ((X[:, :5].sum(1)) > 0).long()
    fn = nn.CrossEntropyLoss()
    outs = {}
    for backend in ("torch", "triton"):
        torch.manual_seed(0)
        net = _net().cuda()
        opt = EcoInt8SGDM([net.fc1, net.fc2], lr=0.5, momentum=0.9, backend=backend)
        for _ in range(20):
            loss = fn(net(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = (net(X).argmax(1) == y).float().mean().item()
        outs[backend] = (loss.item(), acc, opt.last_step_backend)
    lt, at, bt = outs["torch"][0], outs["torch"][1], outs["torch"][2]
    lf, af, bf = outs["triton"][0], outs["triton"][1], outs["triton"][2]
    assert bt == "torch" and bf == "triton"
    assert abs(lf - lt) / max(lt, 1e-6) < 0.15, outs
    assert abs(af - at) < 0.05, outs
