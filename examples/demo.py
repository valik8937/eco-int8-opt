"""Демо: чи виживає навчання, де ВСЕ int8. MLP на синтетиці, поруч FP32 baseline."""
import time
import torch
import torch.nn as nn
from ecoint8 import Int8Linear, Int8SGD

torch.manual_seed(0)
N, D, H = 2048, 20, 64
X = torch.randn(N, D)
y = ((X[:, :5].sum(1)) > 0).long()
loss_fn = nn.CrossEntropyLoss()


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


fp = FP32Net()
fopt = torch.optim.SGD(fp.parameters(), lr=0.05, momentum=0.9)
t0 = time.time()
for ep in range(15):
    fopt.zero_grad()
    loss = loss_fn(fp(X), y)
    loss.backward()
    fopt.step()
fp_time = time.time() - t0
print(f"[FP32] loss={loss.item():.4f} acc={acc_of(fp):.3f} time={fp_time:.2f}s")

net = Int8Net()
opt = Int8SGD([net.fc1, net.fc2], lr=0.05, momentum=0.9)
t0 = time.time()
for ep in range(15):
    out = net(X)
    loss = loss_fn(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (ep + 1) % 5 == 0:
        print(f"  ep={ep+1} loss={loss.item():.4f} acc={acc_of(net):.3f}")
int8_time = time.time() - t0
print(f"[INT8] loss={loss.item():.4f} acc={acc_of(net):.3f} time={int8_time:.2f}s")
print(f"  stats={opt.stats} speedup={fp_time / int8_time:.2f}x")
wbytes = net.fc1.memory_bytes() + net.fc2.memory_bytes()
fbytes = (D * H + H * 2) * 4
print(f"  ваги: int8={wbytes}B vs fp32={fbytes}B (x{fbytes / wbytes:.1f} менше)")
assert acc_of(net) > 0.80, "ЗДОХЛО: точність впала нижче 80%!"
print("НЕ ЗДОХЛО ✔")
