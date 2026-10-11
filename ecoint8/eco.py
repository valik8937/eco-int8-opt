"""ECO (Error-Compensating Optimizer) without master weights.

After each step the updated weights are quantized and the quantization error
is injected into the momentum buffer — an error-feedback loop with zero
extra memory. Based on "ECO: Quantized Training without Full-Precision
Master Weights" (Nikdan et al., Google Research, 2026).

SGDM (paper Algorithm 2)::

    m_tilde = beta * m + (1 - beta) * g
    theta_tilde = theta_hat - lr * m_tilde
    theta_hat = q(theta_tilde)            # stochastic rounding (unbiased, as theory requires)
    e = theta_tilde - theta_hat
    m = m_tilde + alpha * e               # alpha = (1 / lr) * (1 - 1 / beta)

IMPORTANT (from the paper): the weight quantizer must be FIXED — the scales
are frozen at init and never recomputed. A drifting grid breaks the
error-feedback math (consecutive errors would live on different grids).

Requires ``beta > 0`` (otherwise ``alpha`` divides by zero).
"""

from __future__ import annotations

import torch

from .quant import (
    EPS,
    calc_scale,
    quantize_symmetric,
    quantize_tensor_dynamic,
    saturation_ratio,
)

__all__ = ["EcoSGDM", "EcoAdam", "EcoInt8SGDM"]


def _get_grad_w(layer) -> torch.Tensor | None:
    """Fetch the fp32-equivalent weight gradient stashed by the layer's backward."""
    g = layer._grad_w
    if g is None:
        return None
    g = g.clone()
    if not torch.isfinite(g).all():
        return None
    return g


class EcoSGDM:
    """SGD with momentum + ECO error injection. No master weights.

    Stores int8 weights (frozen grid) and an fp32 momentum buffer, which is
    optimizer state — not a master-weight copy. Everything else is transient.

    Note on lr semantics: this uses the ``m = beta*m + (1-beta)*g`` form from
    the paper, so gradient steps are ``(1-beta)``x smaller than torch's SGD.
    Pass ``lr / (1 - beta)`` to match torch SGD step sizes.
    """

    def __init__(
        self,
        int8_layers,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        grad_clip: float | None = 1.0,
        bias_lr: float | None = None,
    ):
        assert momentum > 0, "ECO requires beta > 0 (alpha = (1/lr)(1-1/beta) is undefined otherwise)"
        self.layers = list(int8_layers)
        self.lr = lr
        self.beta = momentum
        self.wd = weight_decay
        self.grad_clip = grad_clip
        self.m = [torch.zeros_like(layer.w_i8, dtype=torch.float32) for layer in self.layers]
        self.s_fixed = [layer.s_w.clone() for layer in self.layers]  # frozen grid
        bparams = [layer.bias for layer in self.layers if layer.bias is not None]
        self._bias_opt = (
            torch.optim.SGD(bparams, lr=bias_lr or lr, momentum=momentum) if bparams else None
        )
        self.step_count = 0
        self.stats = {"skipped": 0, "mean_abs_e": 0.0}

    def set_lr(self, lr: float):
        self.lr = lr
        if self._bias_opt is not None:
            for pg in self._bias_opt.param_groups:
                pg["lr"] = lr

    @torch.no_grad()
    def step(self):
        lr, b = self.lr, self.beta
        alpha = (1.0 / lr) * (1.0 - 1.0 / b)  # ECO injection strength (negative for b < 1)
        self.step_count += 1
        tot_e, n_e = 0.0, 0
        for i, layer in enumerate(self.layers):
            g = _get_grad_w(layer)
            if g is None:
                continue
            if self.grad_clip is not None:
                n = g.norm().item()
                if n > self.grad_clip:
                    g.mul_(self.grad_clip / (n + EPS))
            if self.wd:
                g.add_(layer.w_fp32(), alpha=self.wd)

            m_tilde = self.m[i] * b + g * (1.0 - b)
            th_tilde = layer.w_fp32() - lr * m_tilde

            w_new_i8 = quantize_symmetric(th_tilde, self.s_fixed[i], stochastic=True)
            e = th_tilde - w_new_i8.float() * self.s_fixed[i]

            self.m[i].copy_(m_tilde + alpha * e)

            layer.w_i8.copy_(w_new_i8)
            tot_e += e.abs().mean().item()
            n_e += 1

        if n_e:
            self.stats["mean_abs_e"] = tot_e / n_e
        if self._bias_opt is not None:
            self._bias_opt.step()

    @torch.no_grad()
    def zero_grad(self):
        for layer in self.layers:
            layer._grad_w = None
            if layer.bias is not None and layer.bias.grad is not None:
                layer.bias.grad.zero_()
        if self._bias_opt is not None:
            self._bias_opt.zero_grad()


class EcoAdam:
    """Adam + ECO error injection (paper Algorithm 3). No master weights.

    Stores int8 weights (frozen grid) plus fp32 ``m``/``v`` optimizer states.
    The injection uses Adam's element-wise effective step size::

        m += e * (1 - 1/beta1) / eta_t
    """

    def __init__(
        self,
        int8_layers,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip: float | None = 1.0,
        bias_lr: float | None = None,
    ):
        b1, b2 = betas
        assert b1 > 0, "ECO requires beta1 > 0"
        self.layers = list(int8_layers)
        self.lr = lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.wd = weight_decay
        self.grad_clip = grad_clip
        self.m = [torch.zeros_like(layer.w_i8, dtype=torch.float32) for layer in self.layers]
        self.v = [torch.zeros_like(layer.w_i8, dtype=torch.float32) for layer in self.layers]
        self.s_fixed = [layer.s_w.clone() for layer in self.layers]  # frozen grid
        bparams = [layer.bias for layer in self.layers if layer.bias is not None]
        self._bias_opt = (
            torch.optim.Adam(bparams, lr=bias_lr or lr, betas=betas, eps=eps)
            if bparams
            else None
        )
        self.t = 0
        self.stats = {"skipped": 0, "mean_abs_e": 0.0}

    def set_lr(self, lr: float):
        self.lr = lr
        if self._bias_opt is not None:
            for pg in self._bias_opt.param_groups:
                pg["lr"] = lr

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.b1, self.b2
        bc1, bc2 = 1.0 - b1**self.t, 1.0 - b2**self.t
        tot_e, n_e = 0.0, 0
        for i, layer in enumerate(self.layers):
            g = _get_grad_w(layer)
            if g is None:
                continue
            if self.grad_clip is not None:
                n = g.norm().item()
                if n > self.grad_clip:
                    g.mul_(self.grad_clip / (n + EPS))
            if self.wd:
                g.add_(layer.w_fp32(), alpha=self.wd)

            self.m[i].mul_(b1).add_(g, alpha=1.0 - b1)
            self.v[i].mul_(b2).addcmul_(g, g, value=1.0 - b2)
            v_hat = (self.v[i] / bc2).sqrt() + self.eps
            eta_t = self.lr / (bc1 * v_hat)  # element-wise effective step

            th_tilde = layer.w_fp32() - eta_t * (self.m[i] / bc1)
            w_new_i8 = quantize_symmetric(th_tilde, self.s_fixed[i], stochastic=True)
            e = th_tilde - w_new_i8.float() * self.s_fixed[i]

            self.m[i].add_(e * ((1.0 - 1.0 / b1) / eta_t))

            layer.w_i8.copy_(w_new_i8)
            tot_e += e.abs().mean().item()
            n_e += 1

        if n_e:
            self.stats["mean_abs_e"] = tot_e / n_e
        if self._bias_opt is not None:
            self._bias_opt.step()

    @torch.no_grad()
    def zero_grad(self):
        for layer in self.layers:
            layer._grad_w = None
            if layer.bias is not None and layer.bias.grad is not None:
                layer.bias.grad.zero_()
        if self._bias_opt is not None:
            self._bias_opt.zero_grad()


class EcoInt8SGDM:
    """Fully-int8 ECO: int8 weights, int8 gradients, int8 momentum. Zero fp32 state.

    Neither master weights nor an fp32 momentum buffer (unlike paper ECO).
    fp32 exists only as a transient inside the step; nothing is stored::

        g_hat   = Q_int8(g)              # stochastic rounding (unbiased)
        m_tilde = b * m_hat + (1-b) * g_hat
        th~     = w_hat - lr * m_tilde
        w_new   = Q_fixed(th~)           # frozen grid, see module docstring
        e       = th~ - w_hat_new
        m_new   = m_tilde + alpha * e    # alpha = (1/lr)(1-1/b)
        m_i8    = Q_int8(m_new)

    State per parameter: 1 byte of weights + 1 byte of momentum
    (+ amortized per-row scales). Biases stay fp32 (tiny, not worth it).

    Example:
        model = ecoint8.to_int8(model)          # swap nn.Linear -> Int8Linear
        opt = ecoint8.EcoInt8SGDM(model_int8_layers, lr=0.5, momentum=0.9)
        out = model(x); loss = fn(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    """

    def __init__(
        self,
        int8_layers,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        grad_clip: float | None = 1.0,
        bias_lr: float | None = None,
    ):
        assert momentum > 0, "ECO requires beta > 0"
        self.layers = list(int8_layers)
        self.lr = lr
        self.beta = momentum
        self.wd = weight_decay
        self.grad_clip = grad_clip
        self.m_i8 = [torch.zeros_like(layer.w_i8) for layer in self.layers]
        self.s_m = [1.0 for _ in self.layers]
        self.s_fixed = [layer.s_w.clone() for layer in self.layers]  # frozen weight grid
        bparams = [layer.bias for layer in self.layers if layer.bias is not None]
        self._bias_opt = (
            torch.optim.SGD(bparams, lr=bias_lr or lr, momentum=momentum) if bparams else None
        )
        self.step_count = 0
        self.stats = {"skipped": 0, "dead_grads": 0, "sat_ratio": 0.0, "mean_abs_e": 0.0}

    def set_lr(self, lr: float):
        self.lr = lr
        if self._bias_opt is not None:
            for pg in self._bias_opt.param_groups:
                pg["lr"] = lr

    @torch.no_grad()
    def step(self):
        lr, b = self.lr, self.beta
        alpha = (1.0 / lr) * (1.0 - 1.0 / b)
        self.step_count += 1
        tot_e, n_e, tot_sat = 0.0, 0, 0.0
        for i, layer in enumerate(self.layers):
            g = _get_grad_w(layer)
            if g is None:
                continue
            if self.grad_clip is not None:
                n = g.norm().item()
                if n > self.grad_clip:
                    g.mul_(self.grad_clip / (n + EPS))
            if self.wd:
                g.add_(layer.w_fp32(), alpha=self.wd)

            # 1. gradient into int8 (stochastic = unbiased)
            g_i8, s_g = quantize_tensor_dynamic(g, stochastic=True)
            if bool((g_i8 == 0).all()):  # underflow guard: retry 4x more sensitive
                self.stats["dead_grads"] += 1
                s_g = s_g / 4.0
                g_i8 = quantize_symmetric(g, s_g, stochastic=True)
                if bool((g_i8 == 0).all()):
                    continue
            g_hat = g_i8.float() * s_g

            # 2-3. transient momentum + weight update
            m_tilde = self.m_i8[i].float() * (self.s_m[i] * b) + g_hat * (1.0 - b)
            th_tilde = layer.w_fp32() - lr * m_tilde

            # 4. weights back onto the frozen grid; quantization error
            w_new_i8 = quantize_symmetric(th_tilde, self.s_fixed[i], stochastic=True)
            e = th_tilde - w_new_i8.float() * self.s_fixed[i]

            # 5. ECO injection, momentum back into int8
            m_new = m_tilde + alpha * e
            s_m_new = calc_scale(max(m_new.abs().max().item(), EPS))
            self.m_i8[i].copy_(quantize_symmetric(m_new, s_m_new, stochastic=True))
            self.s_m[i] = s_m_new

            layer.w_i8.copy_(w_new_i8)
            tot_e += e.abs().mean().item()
            tot_sat += saturation_ratio(w_new_i8)
            n_e += 1

        if n_e:
            self.stats["mean_abs_e"] = tot_e / n_e
            self.stats["sat_ratio"] = tot_sat / n_e
        if self._bias_opt is not None:
            self._bias_opt.step()

    @torch.no_grad()
    def zero_grad(self):
        for layer in self.layers:
            layer._grad_w = None
            if layer.bias is not None and layer.bias.grad is not None:
                layer.bias.grad.zero_()
        if self._bias_opt is not None:
            self._bias_opt.zero_grad()

    def memory_bytes(self) -> int:
        """Stored state in bytes over all layers (int8 weights + int8 momentum)."""
        return sum(layer.w_i8.numel() + m.numel() for layer, m in zip(self.layers, self.m_i8))
