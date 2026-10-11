"""eco-int8-opt: strict integer-only ECO training. No fp32 tensors, end to end."""
from .fixedpoint import (
    apply_mult_shift,
    dynamic_scale_from_int,
    int_amax,
    pad_mm_dims,
    quantize_multiplier,
)
from .ilayer import I8Linear, I8ReLU, I8Sequential
from .iopt import I8ECO
from .quant import quantize_symmetric, quantize_tensor_dynamic, quantize_per_channel, dequantize

__version__ = "0.3.0"

__all__ = [
    "I8ECO",
    "I8Linear",
    "I8ReLU",
    "I8Sequential",
    "apply_mult_shift",
    "dynamic_scale_from_int",
    "int_amax",
    "pad_mm_dims",
    "quantize_multiplier",
    "quantize_symmetric",
    "quantize_tensor_dynamic",
    "quantize_per_channel",
    "dequantize",
    "__version__",
]
