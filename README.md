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
- MNIST MLP 784-256-128-10, 5 epochs, cosine decay, stochastic grad
  rounding: strict-int8 trains to ~99%+ train acc with zero weight
  saturation (frozen grids ship with 4x headroom by default, see below).
- State memory: 1 byte/weight + 1 byte/momentum.
- `examples/quad_toy.py` reproduces the ECO paper's Section 3.4 (naive 1/η blow-up
  vs ECO's finite noise floor).

## Honest costs of strict-int8 (read before benchmarking)

- **Stored state is tiny, peak is not.** Stated bytes/param counts cover
  stored tensors only. The step is dominated by int32/int64 transients
  (int32 matmul accumulators, int64 optimizer buffers). On small nets the
  peak can *exceed* fp16: MNIST MLP, 0.4.4 — 32.1 MB peak vs 22.7 MB for
  an fp16 baseline. The win is stored state (2 bytes/param vs 16 for
  fp32 Adam), not peak.
- **Strict steps are slower than fp16**, not faster: eager `torch._int_mm`
  does not hit INT8 tensor cores, and the step carries full-size int32/int64
  transients. Same MNIST run on 0.4.4 (per-row optimizer loop): ~373 s vs
  ~57 s fp16. 0.4.6 vectorized the optimizer (whole-layer tensor ops, two
  host syncs per layer) — expect a multiple of that back on GPU, but not
  fp16 parity. Strict buys "zero fp32 anywhere", not speed — the fused
  Triton backend (`EcoInt8SGDM`, `backend="triton"`) exists for step cost;
  the strict path has no fused backend.
- **Frozen grids must cover the trajectory.** `I8Linear` freezes its
  weight steps and output scale at init/calibration with 4x headroom
  (`headroom=` param). Without it any task whose weights or activations
  grow past init magnitudes saturates the ±127 rails and collapses —
  first the last-layer outputs (logits grow with confidence), then the
  weights. If you see saturation climbing, you need more headroom (at
  the cost of resolution), not a bigger lr.
- **Usage that works:** stochastic grad rounding
  (`quantize_tensor_dynamic(g, stochastic=True)` — deterministic rounding
  injects DC drift into the momentum), a decaying lr schedule, and
  `net.quantize_input(x)` for entry quantization (per-batch amax scales
  silently break the frozen input grid).

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
