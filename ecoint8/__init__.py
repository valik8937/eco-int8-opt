"""eco-int8-opt: int8 ECO optimizers without master weights (plus strict subpackage)."""
from .layers import Int8Linear, to_int8
from .eco import EcoSGDM, EcoAdam, EcoInt8SGDM
from .quant import quantize_symmetric, quantize_tensor_dynamic, quantize_per_channel, dequantize
from .strict import I8ECO, I8Linear, I8ReLU, I8Sequential

__version__ = "0.4.8"

__all__ = [
    "EcoInt8SGDM",
    "EcoSGDM",
    "EcoAdam",
    "I8ECO",
    "I8Linear",
    "I8ReLU",
    "I8Sequential",
    "Int8Linear",
    "to_int8",
    "quantize_symmetric",
    "quantize_tensor_dynamic",
    "quantize_per_channel",
    "dequantize",
    "__version__",
]
