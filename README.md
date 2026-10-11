# eco-int8-opt

Strict integer-only ECO training for PyTorch: **every tensor is int8/int32/int64 —
no fp32 anywhere in the training path.**

Implements the Error-Compensating Optimizer (ECO) idea from *"ECO: Quantized Training without
Full-Precision Master Weights"* (Nikdan et al., Google Research, 2026): after each step the
weights are quantized and the quantization error is injected into the momentum buffer, forming
an error-feedback loop with no extra memory. Here the whole loop — weights, grads, momentum,
matmuls — is integer-only.

## Install

```bash
pip install eco-int8-opt
```

## Quickstart

```python
import torch
import ecoint8

net = ecoint8.I8Sequential(ecoint8.I8Linear(D, H), ecoint8.I8ReLU(), ecoint8.I8Linear(H, 2))
net.calibrate(x_i8, s_x)          # one pass freezes all grids (static quantization)
opt = ecoint8.I8ECO(net.int8_layers(), lr=0.5, momentum=0.9)

logits_i8, s = net.forward_i8(x_i8)   # int8 in -> int8 out, no autograd
# ... loss + grad in fp32 (tiny [N,C] boundary), grad back to int8 ...
net.backward_i8(g_i8, s_g)            # explicit reverse pass, integer only
opt.step(); opt.zero_grad()
```

How it works (`ecoint8.fixedpoint` / `ilayer` / `iopt`):

- int8 matmuls (`torch._int_mm`) with int32 accumulators (CUDA dims padded to spec, bit-exact);
- integer requantization via `(mult, shift)` pairs (TFLite-style), stochastic rounding
  via `torch.randint`, lr/momentum/ECO-injection as fixed-point, weight updates in
  int64 accumulators with fractional bits;
- every kernel input/output is dtype-asserted integral — any fp32 leak explodes immediately;
- `tests/test_strict.py` additionally scans the sources for float-tensor constructors.

Two fp32 boundaries are unavoidable and documented: quantizing the input batch
once (`ecoint8.quant` helpers exist exactly for this), and the final loss (+its grad,
immediately re-quantized). Scales are plain Python floats — scalar bookkeeping,
zero tensor allocations.

## Measured

- MLP, 150 epochs, cosine decay: strict-int8 **0.969** acc vs fp32 0.992
  (`examples/i8_demo.py`).
- State memory: 1 byte/weight + 1 byte/momentum.
- `examples/quad_toy.py` reproduces the ECO paper's Section 3.4 (naive 1/η blow-up
  vs ECO's finite noise floor).

## API

All symbols below are importable from the top level
(`from ecoint8 import I8ECO, I8Sequential, ...`); the strict
integer-only subpackage also lives at `ecoint8.strict`.

| Symbol | What |
|---|---|
| `I8ECO` | Integer-only ECO-SGDM (the optimizer, strict path) |
| `I8Linear` / `I8ReLU` / `I8Sequential` | Integer-only layers + manual pipeline (strict path) |
| `EcoInt8SGDM` | Fully-int8 ECO-SGDM with autograd layers (torch/triton backends) |
| `EcoSGDM` / `EcoAdam` | ECO with fp32 optimizer state, no master weights |
| `Int8Linear` / `to_int8` | int8 matmul layers with fp32 boundary (autograd) |
| `ecoint8.fixedpoint` | `quantize_multiplier`, `apply_mult_shift`, `pad_mm_dims`, ... |
| `ecoint8.quant` | Boundary helpers (entry/exit quantization only) |

## Citation

The error-feedback core of this package implements the method from:

> Mahdi Nikdan, Amir Zandieh, Dan Alistarh, Vahab Mirrokni.
> **ECO: Quantized Training without Full-Precision Master Weights.**
> Google Research, 2026. [arXiv:2601.22101](https://arxiv.org/abs/2601.22101)

```bibtex
@misc{nikdan2026eco,
  title={ECO: Quantized Training without Full-Precision Master Weights},
  author={Mahdi Nikdan and Amir Zandieh and Dan Alistarh and Vahab Mirrokni},
  year={2026},
  eprint={2601.22101},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## License

MIT
