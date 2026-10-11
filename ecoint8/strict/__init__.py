"""Strict integer-only ECO training (experimental subpackage).

No fp32 tensors anywhere in the training path: int8/int32/int64 only,
manual forward/backward (no autograd), frozen quantization grids.
See ``examples/i8_demo.py``.
"""
from .ilayer import I8Linear, I8ReLU, I8Sequential
from .iopt import I8ECO

__all__ = ["I8ECO", "I8Linear", "I8ReLU", "I8Sequential"]
