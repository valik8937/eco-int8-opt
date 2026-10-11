"""ECO на реальному MLP: FP32 з master weights vs naive int8 vs ECO, cosine decay LR -> 0.

Чесне порівняння (як у статті, Appendix A): всі три в одній семантиці кроків.
torch SGD і naive беруть повний g; ECO у формі m = b*m + (1-b)*g, тому йому
lr множимо на 1/(1-b), щоб ефективні кроки градієнта збігалися. Теорія ECO
від eta не залежить — це просто інше eta.

Довге навчання (150 епох) із затуханням: тут naive без master weights має
стагнувати в хвості (помилка ~1/eta), а ECO з інжекцією триматися біля MW.
"""
import math
import torch
import torch.nn as nn
from ecoint8 import Int8Linear, Int8SGD
from ecoint8.eco import EcoSGDM, EcoInt8SGDM
from ecoint8.quant import quantize_per_channel

torch.manual_seed(0)
N, D, H = 2048, 20, 32
X = torch.randn(N, D)
y = ((X[:, :5].sum(1)) > 0).long()
loss_fn = nn.CrossEntropyLoss()
EPOCHS, LR0, BETA = 150, 0.05, 0.9


def cosine_lr(t: int) -> float:
    return LR0 * 0.5 * (1 + math.cos(math.pi * t / EPOCHS)) + 1e-4


class FP32Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D, H)
        self.fc2 = nn.Linear(H, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Int8Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Int8Linear(D, H)
        self.fc2 = Int8Linear(H, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def acc_of(net):
    with torch.no_grad():
        return (net(X).argmax(1) == y).float().mean().item()


def loss_of(net):
    with torch.no_grad():
        return loss_fn(net(X), y).item()


def clone_weights(src, dst):
    for a, b in zip([src.fc1, src.fc2], [dst.fc1, dst.fc2]):
        w, s = quantize_per_channel(a.weight.detach(), axis=0)
        b.w_i8.copy_(w)
        b.s_w.copy_(s)
        if b.bias is not None:
            b.bias.data.copy_(a.bias.detach())


ref = FP32Net()
fp = FP32Net()
fp.load_state_dict(ref.state_dict())
fopt = torch.optim.SGD(fp.parameters(), lr=LR0, momentum=BETA)

nv = Int8Net()
clone_weights(ref, nv)
nopt = Int8SGD([nv.fc1, nv.fc2], lr=LR0, momentum=BETA)

ec = Int8Net()
clone_weights(ref, ec)
eopt = EcoSGDM([ec.fc1, ec.fc2], lr=LR0 / (1 - BETA), momentum=BETA)  # вирівняні кроки

f8 = Int8Net()
clone_weights(ref, f8)
f8opt = EcoInt8SGDM([f8.fc1, f8.fc2], lr=LR0 / (1 - BETA), momentum=BETA)  # ФУЛ int8 ECO

print(f"{'ep':>4} {'lr':>8} | {'fp32':>13} {'naive':>13} {'ECO':>13} {'FULL-int8':>13}")
for ep in range(EPOCHS):
    lr = cosine_lr(ep)
    for pg in fopt.param_groups:
        pg["lr"] = lr
    nopt.lr = lr
    eopt.set_lr(lr / (1 - BETA))
    f8opt.set_lr(lr / (1 - BETA))

    fopt.zero_grad()
    loss_fn(fp(X), y).backward()
    fopt.step()

    nopt.zero_grad()
    loss_fn(nv(X), y).backward()
    nopt.step()

    eopt.zero_grad()
    loss_fn(ec(X), y).backward()
    eopt.step()

    f8opt.zero_grad()
    loss_fn(f8(X), y).backward()
    f8opt.step()

    if (ep + 1) % 25 == 0 or ep == 0:
        print(f"{ep+1:>4} {lr:>8.5f} | {loss_of(fp):>7.4f}/{acc_of(fp):.3f} "
              f"{loss_of(nv):>7.4f}/{acc_of(nv):.3f} {loss_of(ec):>7.4f}/{acc_of(ec):.3f} "
              f"{loss_of(f8):>7.4f}/{acc_of(f8):.3f}")

print(f"\nфінал: fp32 acc={acc_of(fp):.3f} | naive acc={acc_of(nv):.3f} {nopt.stats} | "
      f"ECO acc={acc_of(ec):.3f} mean|e|={eopt.stats['mean_abs_e']:.2e} | "
      f"FULL-int8 acc={acc_of(f8):.3f} {f8opt.stats}")
print(f"пам'ять стану: FULL-int8={f8opt.memory_bytes()}B vs "
      f"fp32-SGD={(sum(p.numel() for p in fp.parameters()) * 4 * 2)}B (ваги+моментум fp32)")
