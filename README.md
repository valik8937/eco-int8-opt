# eco-int8

Fully-int8 ECO optimizers for PyTorch: **int8 weights, int8 gradients, int8 momentum — zero master weights.**

Implements the Error-Compensating Optimizer (ECO) idea from *"ECO: Quantized Training without
Full-Precision Master Weights"* (Nikdan et al., Google Research, 2026): after each step the
weights are quantized and the quantization error is injected into the momentum buffer, forming
an error-feedback loop with no extra memory. Here even the momentum lives in int8.

## Install

```bash
pip install eco-int8
```

## Quickstart

```python
import torch
import torch.nn as nn
import ecoint8

# 1. swap nn.Linear -> int8 layers (weights stored as int8 + per-row scales)
model = ecoint8.to_int8(model)

# 2. optimize — same call pattern as torch SGD
int8_layers = [m for m in model.modules() if isinstance(m, ecoint8.Int8Linear)]
opt = ecoint8.EcoInt8SGDM(int8_layers, lr=0.5, momentum=0.9)

for x, y in loader:
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
```

Notes:

- `Int8Linear` computes forward/backward matmuls in int8 (`torch._int_mm`) and stashes
  weight grads for the optimizer — `loss.backward(); opt.step()` just works.
- The weight quantization grid is **frozen** at optimizer construction (required by ECO theory).
  Use stochastic rounding (default) — deterministic rounding breaks the guarantees.
- Lr semantics follow the paper (`m = βm + (1−β)g`), so pass `lr / (1 − beta)` to match
  torch SGD step sizes.
- `opt.memory_bytes()` reports stored optimizer state in bytes (1B/weight + 1B/momentum).

## API

| Symbol | What |
|---|---|
| `EcoInt8SGDM` | Fully-int8 ECO (the main event) |
| `I8ECO` + `I8Linear` / `I8Sequential` | **Strict** integer-only path: zero fp32 tensors (see below) |
| `EcoSGDM` / `EcoAdam` | ECO with fp32 momentum states (paper-faithful) |
| `Int8SGD` | All-int8 SGD without error feedback (naive baseline) |
| `Int8Linear` / `to_int8` | int8 linear layer + model converter |
| `to_fbgemm` | Export trained layers to prepacked FBGEMM for ~2x faster CPU inference |

## Strict integer-only path (no fp32 tensors, end to end)

`ecoint8.fixedpoint` / `ilayer` / `iopt` implement training where **every tensor**
is int8/int32/int64: int8 matmuls (`torch._int_mm`) with int32 accumulators,
integer requantization via `(mult, shift)` pairs (TFLite-style), stochastic
rounding via `torch.randint`, lr/momentum/ECO-injection as fixed-point.
Runtime dtype asserts guard every kernel boundary, and
`tests/test_strict.py` scans the sources for float-tensor constructors.

```python
net = ecoint8.I8Sequential(ecoint8.I8Linear(D, H), ecoint8.I8ReLU(), ecoint8.I8Linear(H, 2))
net.calibrate(x_i8, s_x)          # one pass freezes all grids (static quantization)
opt = ecoint8.I8ECO(net.int8_layers(), lr=0.5, momentum=0.9)

logits_i8, s = net.forward_i8(x_i8)   # int8 in -> int8 out, no autograd
# ... loss + grad in fp32 (tiny [N,C] boundary), grad back to int8 ...
net.backward_i8(g_i8, s_g)            # explicit reverse pass, integer only
opt.step(); opt.zero_grad()
```

Two fp32 boundaries are unavoidable and documented: quantizing the input batch
once, and the final loss (+its grad, immediately re-quantized). Scales are
plain Python floats — scalar bookkeeping, zero tensor allocations.
Measured: strict-int8 0.969 acc vs 0.985 mixed ECO vs 0.992 fp32
(`examples/i8_demo.py`).

## Measured

- MLP, 150 epochs, cosine decay: fp32 0.992 acc / ECO-int8 0.985 / naive-int8 0.976
  (naive degrades in the decayed-lr tail, ECO holds — see `examples/eco_demo.py`).
- State memory: 1408B vs 5904B for fp32 SGD with momentum (~4x less).
- `examples/quad_toy.py` reproduces the paper's Section 3.4 (naive 1/η blow-up vs ECO floor).

## License

MIT
