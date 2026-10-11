"""eco-int8: fully-int8 ECO optimizers without master weights."""
from .quant import quantize_symmetric, quantize_tensor_dynamic, quantize_per_channel, dequantize
from .layers import Int8Linear, to_int8
from .optimizer import Int8SGD
from .eco import EcoSGDM, EcoAdam, EcoInt8SGDM
from .deploy import FbgemmInt8Linear, to_fbgemm
from .ilayer import I8Linear, I8ReLU, I8Sequential
from .iopt import I8ECO

__version__ = "0.2.0"

__all__ = [
    "EcoInt8SGDM",
    "I8ECO",
    "EcoSGDM",
    "EcoAdam",
    "Int8SGD",
    "Int8Linear",
    "I8Linear",
    "I8ReLU",
    "I8Sequential",
    "to_int8",
    "FbgemmInt8Linear",
    "to_fbgemm",
    "quantize_symmetric",
    "quantize_tensor_dynamic",
    "quantize_per_channel",
    "dequantize",
    "__version__",
]
