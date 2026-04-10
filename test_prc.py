"""Test Prototype-Routed Computation vs Standard Transformer."""
import torch, torch.nn as nn, torch.nn.functional as F, math, time

device = 'cuda'

class PrototypeRoutedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, n_proto=32, rank=16):
        super().__init__()
        self.in_dim, self.out_dim, self.n_proto, self.rank = in_dim, out_dim, n_proto, rank
        self.prototypes = nn.Parameter(torch.randn(n_proto, in_dim) * 0.02)
        self.B = nn.Parameter(torch.randn(n_proto, rank, in_dim) * 0.02)
        self.A = nn.Parameter(torch.randn(n_proto, out_dim, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_proto, out_dim))
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        shape = x.shape[:-1]
        xf = x.reshape(-1, self.in_dim)
        B = xf.shape[0]
        dists = torch.cdist(xf.unsqueeze(0), self.prototypes.unsqueeze(0)).squeeze(0)
        w = F.softmax(-dists / self.temp.abs().clamp(min=0.1), dim=1)
        topw, topi = w.topk(2, dim=1)
        topw = topw / topw.sum(dim=1, keepdim=True)
        out = torch.zeros(B, self.out_dim, device=x.device)
        for t in range(2):
            idx = topi[:, t]
            wt = topw[:, t:t+1]
            h = torch.bmm(self.B[idx], xf.unsqueeze(2)).squeeze(2)
            y = torch.bmm(self.A[idx], h.unsqueeze(2)).squeeze(2) + self.bias[idx]
            out += wt * y
        return out.reshape(*shape, self.out_dim)

class PRCLayer(nn.Module):
    def __init__(self, d, nh, ff, np=32, r=16):
        super().__init__()
        self.nh, self.hd = nh, d // nh
        self.q = PrototypeRoutedLinear(d, d, np, r)
        self.k = PrototypeRoutedLinear(d, d, np, r)
        self.v = PrototypeRoutedLinear(d, d, np, r)
        self.o = PrototypeRoutedLinear(d, d, np, r)
        self.gate = PrototypeRoutedLinear(d, ff, np, r)
        self.up = PrototypeRoutedLinear(d, ff, np, r)
        self.down = PrototypeRoutedLinear(ff, d, np, r)
        self.n1 = nn.RMSNorm(d)
        self.n2 = nn.RMSNorm(d)

    def forward(self, x):
        B, T, C = x.shape
        h = self.n1(x)
        q = self.q(h).reshape(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).reshape(B, T, self.nh, self.hd).transpose(1, 2)
        v = self.v(h).reshape(B, T, self.nh, self.hd).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        if T > 1:
            a = a.masked_fill(torch.triu(torch.ones(T, T, device=x.device), 1).bool(), float('-inf'))
        o = (F.softmax(a, -1) @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.o(o)
        h = self.n2(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x

class PRCModel(nn.Module):
    def __init__(self, V, d, nh, ff, nl, np=32, r=16):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.layers = nn.ModuleList([PRCLayer(d, nh, ff, np, r) for _ in range(nl)])
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for l in self.layers:
            h = l(h)
        return self.head(self.norm(h))

class StdModel(nn.Module):
    def __init__(self, V, d, nh, ff, nl):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.nh, self.hd = nh, d // nh
        self.layers = nn.ModuleList()
        for _ in range(nl):
            self.layers.append(nn.ModuleDict({
                'n1': nn.RMSNorm(d), 'n2': nn.RMSNorm(d),
                'qkv': nn.Linear(d, 3 * d), 'o': nn.Linear(d, d),
                'g': nn.Linear(d, ff), 'u': nn.Linear(d, ff), 'dn': nn.Linear(ff, d),
            }))
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, x):
        h = self.emb(x)
        for l in self.layers:
            B, T, C = h.shape
            n = l['n1'](h)
            qkv = l['qkv'](n).reshape(B, T, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
            a = (qkv[0] @ qkv[1].transpose(-2, -1)) / math.sqrt(self.hd)
            if T > 1:
                a = a.masked_fill(torch.triu(torch.ones(T, T, device=x.device), 1).bool(), float('-inf'))
            o = (F.softmax(a, -1) @ qkv[2]).transpose(1, 2).reshape(B, T, C)
            h = h + l['o'](o)
            n = l['n2'](h)
            h = h + l['dn'](F.silu(l['g'](n)) * l['u'](n))
        return self.head(self.norm(h))


V, d, nh, ff, nl = 100, 128, 4, 256, 4
seq_len = 16

print("=== PROTOTYPE-ROUTED COMPUTATION vs STANDARD TRANSFORMER ===")
print("Task: next token = (current + 3) mod 100")
print()

torch.manual_seed(42)
def make_data(n):
    x = torch.randint(1, V, (n, seq_len), device=device)
    y = (x + 3) % V
    return x, y

train_x, train_y = make_data(2000)
test_x, test_y = make_data(500)

for np_val, r_val, label in [(16, 8, "PRC-16p-8r"), (32, 16, "PRC-32p-16r"), (64, 32, "PRC-64p-32r")]:
    prc = PRCModel(V, d, nh, ff, nl, np=np_val, r=r_val).to(device)
    pp = sum(p.numel() for p in prc.parameters())
    opt = torch.optim.Adam(prc.parameters(), lr=0.001)
    for ep in range(500):
        idx = torch.randint(0, 2000, (128,))
        loss = F.cross_entropy(prc(train_x[idx]).reshape(-1, V), train_y[idx].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        tl = F.cross_entropy(prc(test_x).reshape(-1, V), test_y.reshape(-1))
        ta = (prc(test_x).argmax(-1) == test_y).float().mean()
    print(f"  {label:20s}: {pp:>8,} params  loss={tl:.4f}  acc={ta:.4f}")

std = StdModel(V, d, nh, ff, nl).to(device)
sp = sum(p.numel() for p in std.parameters())
opt = torch.optim.Adam(std.parameters(), lr=0.001)
for ep in range(500):
    idx = torch.randint(0, 2000, (128,))
    loss = F.cross_entropy(std(train_x[idx]).reshape(-1, V), train_y[idx].reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
with torch.no_grad():
    tl = F.cross_entropy(std(test_x).reshape(-1, V), test_y.reshape(-1))
    ta = (std(test_x).argmax(-1) == test_y).float().mean()
print(f"  {'Standard':20s}: {sp:>8,} params  loss={tl:.4f}  acc={ta:.4f}")

prc_base = PRCModel(V, d, nh, ff, nl, 32, 16)
print(f"\n  PRC params:  {sum(p.numel() for p in prc_base.parameters()):,}")
print(f"  Std params:  {sp:,}")
print(f"  Ratio:       {sp / sum(p.numel() for p in prc_base.parameters()):.1f}x")
