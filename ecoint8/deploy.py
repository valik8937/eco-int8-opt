"""Експорт Int8Linear в швидкий FBGEMM inference (prepacked int8 weight, ~2x на CPU).

Тренуємо через Int8Linear (custom autograd), деплоїмо через FbgemmInt8Linear.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class FbgemmInt8Linear(nn.Module):
    def __init__(self, w_i8: torch.Tensor, s_w: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        # FBGEMM хоче per-channel affine qint8 + fp32 bias
        n = w_i8.shape[0]
        scales = s_w.reshape(n).double()
        zp = torch.zeros(n, dtype=torch.long)
        # в FBGEMM кладемо ДЕквантовані ваги (w_i8*s_w) + ті ж скейли — round-trip точний
        wq = torch.quantize_per_channel(w_i8.float() * s_w.float(), scales, zp, axis=0, dtype=torch.qint8)
        self.Wp = torch.ops.quantized.linear_prepack(wq, bias.float() if bias is not None else torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_dynamic(x, self.Wp)


def to_fbgemm(int8_layer) -> FbgemmInt8Linear:
    return FbgemmInt8Linear(int8_layer.w_i8.cpu(), int8_layer.s_w.cpu(),
                            int8_layer.bias.detach().cpu() if int8_layer.bias is not None else None)
