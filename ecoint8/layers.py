"""Int8Linear: ваги int8 + активації int8, matmul через torch._int_mm (int8->int32).

Схема forward:
  X_f32 --quant--> X_i8 (scale sx, динамічний per-tensor)
  W зберігається як W_i8 + scale sw (per-output-channel)
  acc_i32 = _int_mm(X_i8, W_i8.T)      <- реальний int8 matmul, швидкий на x86/VNNI
  Y = acc_f32 * sx * sw + bias

Backward теж через int8 matmul:
  G = grad_output квантований в int8 (scale sg)
  dW_acc = _int_mm(G_i8, X_i8)  -> точний (обидва скейли per-tensor), ховається в layer._grad_w
  dX_acc = _int_mm(G_i8, W_i8)  -> наближений: per-channel sw замінено середнім (інакше не факторизується
                                  з int32 суми; похибка масштабу в backward для навчання ок)

w_i8 — buffer (int тензори не мають autograd), тому grad ваг передається оптимізатору
через side-channel layer._grad_w. loss.backward() -> opt.step() працює як зазвичай.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import weakref
from .quant import quantize_tensor_dynamic, quantize_per_channel, dequantize


def _pad8(n: int) -> int:
    return (n + 7) // 8 * 8


def _int_mm(a8: torch.Tensor, b8: torch.Tensor) -> torch.Tensor:
    """Безпечна обгортка над torch._int_mm. Повертає int32.

    На CUDA _int_mm вимагає розміри кратні 8 — добиваємо нулями і обрізаємо
    результат. На нулях добуток той самий, операція точна.
    """
    a8 = a8.contiguous()
    b8 = b8.contiguous()
    if a8.is_cuda:
        M, K = a8.shape
        N = b8.shape[1]
        M8, K8, N8 = _pad8(M), _pad8(K), _pad8(N)
        if (M8, K8, N8) != (M, K, N):
            a8 = torch.nn.functional.pad(a8, (0, K8 - K, 0, M8 - M))
            b8 = torch.nn.functional.pad(b8, (0, N8 - N, 0, K8 - K))
            return torch._int_mm(a8, b8)[:M, :N]
    return torch._int_mm(a8, b8)


class Int8LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_f32, w_i8, s_w, bias, layer_ref):
        x_i8, s_x = quantize_tensor_dynamic(x_f32)
        acc = _int_mm(x_i8, w_i8.t())  # [N, out] int32
        y = acc.to(torch.float32) * s_x * s_w.t()
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(x_i8, w_i8, s_w)
        ctx.s_x = float(s_x)
        ctx.has_bias = bias is not None
        ctx.layer_ref = layer_ref  # weakref на модуль для передачі grad_w
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x_i8, w_i8, s_w = ctx.saved_tensors
        s_x = ctx.s_x
        g_i8, s_g = quantize_tensor_dynamic(grad_out)
        # dW: точний, обидва скейли per-tensor
        dw_acc = _int_mm(g_i8.t(), x_i8)  # [out, in] int32
        grad_w = dw_acc.to(torch.float32) * (s_g * s_x)
        # dX: наближений (середній sw замість per-channel)
        dx_acc = _int_mm(g_i8, w_i8)  # [N, in] int32
        grad_x = dx_acc.to(torch.float32) * (s_g * s_w.mean().item())
        grad_bias = grad_out.sum(0) if ctx.has_bias else None
        # side-channel: кладемо grad ваг на шар
        layer = ctx.layer_ref()
        if layer is not None:
            layer._grad_w = grad_w
        return grad_x, None, None, grad_bias, None


class Int8Linear(nn.Module):
    """Drop-in заміна nn.Linear з int8 вагами і int8 обчисленнями."""

    def __init__(self, in_f: int, out_f: int, bias: bool = True):
        super().__init__()
        w_fp = torch.randn(out_f, in_f) * (2.0 / in_f) ** 0.5
        w_i8, s_w = quantize_per_channel(w_fp, axis=0)
        self.register_buffer("w_i8", w_i8)
        self.register_buffer("s_w", s_w)  # [out,1]
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self._grad_w: torch.Tensor | None = None

    @classmethod
    def from_float(cls, lin: nn.Linear) -> "Int8Linear":
        m = cls(lin.in_features, lin.out_features, lin.bias is not None)
        w_i8, s_w = quantize_per_channel(lin.weight.detach(), axis=0)
        m.w_i8.copy_(w_i8)
        m.s_w.copy_(s_w)
        if lin.bias is not None:
            m.bias.data.copy_(lin.bias.detach())
        return m

    def forward(self, x):
        return Int8LinearFunc.apply(x, self.w_i8, self.s_w, self.bias, weakref.ref(self))

    def extra_repr(self):
        return f"in={self.w_i8.shape[1]}, out={self.w_i8.shape[0]}, int8=True"

    def w_fp32(self) -> torch.Tensor:
        """Транзієнтний fp32 вид ваг (тільки для обчислень, не зберігається)."""
        return dequantize(self.w_i8, self.s_w)

    def memory_bytes(self) -> int:
        return self.w_i8.numel()  # 1 байт на вагу проти 4 у fp32


def to_int8(model: nn.Module) -> nn.Module:
    """Рекурсивно міняє всі nn.Linear на Int8Linear (з копією ваг). Bias лишається fp32."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            setattr(model, name, Int8Linear.from_float(child))
        else:
            to_int8(child)
    return model
