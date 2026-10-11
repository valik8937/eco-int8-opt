"""Іграшка з Section 3.4 / Appendix C статті: f(x) = L/2 * x^2.

Ідеалізована модель статті: q(x) = x + xi, xi — незалежний нуль-середній шум,
Var = s^2 (навіть у нулі шум не зникає — тому є стаціонарний розподіл).
Порівнюємо стаціонарну E[(grad)^2] при eta -> 0:
  MW    -> L^2 s^2              (константа)
  naive -> ~ L s^2 / (2 eta)     (ВИБУХ)
  ECO   -> L^2 s^2 / (1-b^2)     (скінченна полиця)

Запуск: python3 quad_toy.py
"""
import torch

torch.manual_seed(0)
L, DELTA, BETA = 1.0, 0.1, 0.9
SIG2 = DELTA ** 2 / 12          # Var(U(-D/2, D/2))
BATCH, T, TAIL = 400, 30000, 5000


def q_noise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q(x) = x + xi; повертає (x_hat, e = x - x_hat)."""
    xi = (torch.rand_like(x) - 0.5) * DELTA
    return x + xi, -xi


def run(mode: str, eta: float) -> float:
    x = torch.randn(BATCH) * 2
    m = torch.zeros(BATCH)
    if mode in ("naive", "eco"):
        x, _ = q_noise(x)
    alpha = (1.0 / eta) * (1.0 - 1.0 / BETA)
    acc, cnt = 0.0, 0
    for t in range(T):
        if mode == "mw":
            xh, _ = q_noise(x)
            m = BETA * m + (1 - BETA) * L * xh
            x = x - eta * m
            metric_x = xh
        else:
            m = BETA * m + (1 - BETA) * L * x
            xt = x - eta * m
            x, e = q_noise(xt)
            if mode == "eco":
                m = m + alpha * e
            metric_x = x
        if t >= T - TAIL:
            acc += ((L * metric_x) ** 2).mean().item()
            cnt += 1
    return acc / cnt


if __name__ == "__main__":
    print(f"теорія: MW -> {L**2*SIG2:.5f}, ECO -> {L**2*SIG2/(1-BETA**2):.5f}, "
          f"naive(eta=0.002) -> {L*SIG2/(2*0.002):.3f}")
    print(f"{'eta':>7} {'MW':>10} {'naive':>10} {'ECO':>10}")
    for eta in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
        print(f"{eta:>7} {run('mw', eta):>10.5f} {run('naive', eta):>10.5f} {run('eco', eta):>10.5f}")
    print("\nОчікування: MW -> const, naive -> вибух ~1/eta, ECO -> скінченна полиця.")
