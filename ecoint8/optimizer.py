"""Int8SGD: ВСЕ в int8 — ваги, градієнти, моментум.

Зберігається в int8: w_i8+s_w (шар), g_i8+s_g (транзієнт), m_i8+s_m (моментум).
Жодної fp32 мастер-копії ваг. fp32 існує тільки як транзієнт всередині кроку.

Крок: m = mu*m + g  ->  w -= lr*m, все з стохастичним округленням назад в int8.

Анти-смерть (щоб не здохло):
  - grad_boost: grad_i8 всі нулі (underflow) -> чутливість x4, інакше крок пропускається
  - EMA по amax градієнта -> стабільний s_g без стрибків
  - saturation guard: ваги вперлись в ±127 (>10%) -> scale x2
  - стохастичне округлення оновлення — інакше малі lr*grad завжди дають 0 і навчання стоїть
  - NaN/Inf детект — крок пропускається, лічильник skipped

Використання:
  out = model(X); loss = fn(out, y); loss.backward(); opt.step(); opt.zero_grad()
  (bias-гради йдуть звичайним autograd + окремим fp32 SGD; їх веде внутрішній оптимізатор)
"""
from __future__ import annotations
import torch
from .quant import quantize_symmetric, calc_scale, saturation_ratio, is_dead_grad, EPS


class Int8SGD:
    def __init__(self, int8_layers, lr: float = 1e-2, momentum: float = 0.9,
                 weight_decay: float = 0.0, grad_clip: float | None = 1.0,
                 bias_lr: float | None = None):
        self.layers = list(int8_layers)
        self.lr = lr
        self.momentum = momentum
        self.wd = weight_decay
        self.grad_clip = grad_clip
        self.m_i8 = [torch.zeros_like(l.w_i8) for l in self.layers]
        self.s_m = [1.0 for _ in self.layers]
        self.g_ema = [None for _ in self.layers]
        # fp32 bias-оптимізатор (bias дрібні, їх в int8 ганяти нема сенсу)
        bparams = [l.bias for l in self.layers if l.bias is not None]
        self._bias_opt = torch.optim.SGD(bparams, lr=bias_lr or lr, momentum=momentum) if bparams else None
        self.step_count = 0
        self.stats = {"dead_grads": 0, "rescales": 0, "skipped": 0}

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        for i, layer in enumerate(self.layers):
            g = layer._grad_w
            if g is None:
                continue
            if not torch.isfinite(g).all():
                self.stats["skipped"] += 1
                continue
            g = g.clone()
            if self.grad_clip is not None:
                n = g.norm().item()
                if n > self.grad_clip:
                    g.mul_(self.grad_clip / (n + EPS))
            if self.wd:
                g.add_(layer.w_fp32(), alpha=self.wd)

            # --- 1. динамічний скейл градієнта (EMA по amax) ---
            amax = g.abs().max().item()
            self.g_ema[i] = amax if self.g_ema[i] is None else 0.9 * self.g_ema[i] + 0.1 * amax
            s_g = calc_scale(max(self.g_ema[i], EPS))

            # --- 2. градієнт в int8 (стохастично) ---
            g_i8 = quantize_symmetric(g, s_g, stochastic=True)

            # --- анти-смерть #1: underflow-буст ---
            if is_dead_grad(g_i8):
                self.stats["dead_grads"] += 1
                s_g = s_g / 4.0
                g_i8 = quantize_symmetric(g, s_g, stochastic=True)
                if is_dead_grad(g_i8):
                    continue  # реально нульовий град — пропускаємо

            # --- 3. momentum в int8: m = mu*m + g ---
            m_fp = self.m_i8[i].float() * (self.s_m[i] * self.momentum) + g_i8.float() * s_g
            s_m_new = calc_scale(max(m_fp.abs().max().item(), EPS))
            self.m_i8[i].copy_(quantize_symmetric(m_fp, s_m_new, stochastic=True))
            self.s_m[i] = s_m_new

            # --- 4. оновлення ваг, re-quant в той же s_w ---
            s_w = layer.s_w
            w_new = layer.w_i8.float() * s_w - self.m_i8[i].float() * (self.lr * self.s_m[i])
            layer.w_i8.copy_(torch.clamp(
                torch.floor(w_new / s_w + 0.5 + torch.rand_like(w_new)), -127, 127).to(torch.int8))

            # --- анти-смерть #2: saturation guard ---
            if saturation_ratio(layer.w_i8) > 0.10:
                layer.s_w.mul_(2.0)
                layer.w_i8.copy_(torch.clamp(torch.round(w_new / layer.s_w), -127, 127).to(torch.int8))
                self.stats["rescales"] += 1

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
