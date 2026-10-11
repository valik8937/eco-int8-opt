"""eco-int8-opt: int8 ECO optimizers without master weights (plus strict subpackage)."""
from .layers import Int8Linear, to_int8
from .eco import EcoSGDM, EcoAdam, EcoInt8SGDM
from .quant import quantize_symmetric, quantize_tensor_dynamic, quantize_per_channel, dequantize

__version__ = "0.4.2"

__all__ = [
    "EcoInt8SGDM",
    "EcoSGDM",
    "EcoAdam",
    "Int8Linear",
    "to_int8",
    "quantize_symmetric",
    "quantize_tensor_dynamic",
    "quantize_per_channel",
    "dequantize",
    "__version__",
]
