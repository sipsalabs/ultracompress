"""Portfolio-ready plots from results.json:
  claim16_envelope.png  - dual bar chart (PPL ratio + T10 teacher-agreement)
  claim16_bpw.png       - bpw convergence at 2.40 target
"""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("results.json") as f:
    data = json.load(f)

rows = data["models"]
names = [r["model"] for r in rows]
params = [r["params_B"] for r in rows]
ratios = [r["ratio"] for r in rows]
agrs  = [r.get("t10_agreement", 0) for r in rows]
rets  = [r.get("t10_retention", 0) for r in rows]
bpws  = [r.get("bpw", 2.4) for r in rows]
fams  = [r["family"] for r in rows]

# sort by param count
order = np.argsort(params)
names   = [names[i]  for i in order]
params  = [params[i] for i in order]
ratios  = [ratios[i] for i in order]
agrs    = [agrs[i]   for i in order]
rets    = [rets[i]   for i in order]
bpws    = [bpws[i]   for i in order]
fams    = [fams[i]   for i in order]

FAMILY_COLORS = {"Llama-2":"#4C78A8", "Qwen3":"#F58518", "Mistral":"#54A24B"}
colors = [FAMILY_COLORS[f] for f in fams]

# --- envelope plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
x = np.arange(len(names))

bars1 = ax1.bar(x, ratios, color=colors, edgecolor="#222", linewidth=0.8)
ax1.axhline(1.0, color="#888", linestyle="--", linewidth=0.9, label="fp16 baseline")
ax1.axhline(1.9, color="#d62728", linestyle=":", linewidth=1.0, label="envelope ceiling (1.9×)")
ax1.set_ylabel("PPL ratio (v17 / fp16)", fontsize=11)
ax1.set_ylim(0.9, 2.0)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{n}\n({p}B)" for n, p in zip(names, params)], fontsize=9)
for i, (r, n) in enumerate(zip(ratios, names)):
    ax1.text(i, r + 0.02, f"{r:.3f}×", ha="center", fontsize=9, fontweight="bold")
ax1.set_title(f"Claim 16: PPL ratio at 2.40 bpw — {len(names)} models, zero retuning", fontsize=11)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(axis="y", alpha=0.3)

bars2 = ax2.bar(x, agrs, color=colors, edgecolor="#222", linewidth=0.8, label="T10 teacher-agreement")
ax2.bar(x, rets, color=colors, edgecolor="#222", linewidth=0.8, alpha=0.35, label="T10 retention")
ax2.axhline(93.0, color="#d62728", linestyle=":", linewidth=1.0, label="envelope floor (93%)")
ax2.set_ylabel("%", fontsize=11)
ax2.set_ylim(85, 100)
ax2.set_xticks(x)
ax2.set_xticklabels([f"{n}\n({p}B)" for n, p in zip(names, params)], fontsize=9)
for i, a in enumerate(agrs):
    ax2.text(i, a + 0.3, f"{a:.2f}%", ha="center", fontsize=9, fontweight="bold")
ax2.set_title("Claim 16: top-10 fidelity at 2.40 bpw (n=500 wikitext103)", fontsize=11)
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(axis="y", alpha=0.3)

# family legend
from matplotlib.patches import Patch
leg = [Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
fig.legend(handles=leg, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)

fig.suptitle("Cross-family validation of 2.40-bpw compression law  |  α_attn=0.25, α_mlp=0.125",
             fontsize=12, fontweight="bold", y=1.00)
fig.tight_layout()
fig.savefig("claim16_envelope.png", dpi=160, bbox_inches="tight")
print("Saved claim16_envelope.png")

# --- bpw convergence plot ---
fig, ax = plt.subplots(figsize=(7, 4.0))
ax.bar(x, bpws, color=colors, edgecolor="#222", linewidth=0.8)
ax.axhline(2.40, color="#d62728", linestyle="--", linewidth=1.0, label="target 2.40 bpw")
for i, b in enumerate(bpws):
    ax.text(i, b + 0.003, f"{b:.4f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("bits per weight", fontsize=11)
ax.set_ylim(2.38, 2.42)
ax.set_xticks(x)
ax.set_xticklabels([f"{n}\n({p}B)" for n, p in zip(names, params)], fontsize=9)
ax.set_title(f"Body bpw across {len(names)} models — 0.4% spread at 2.40 target", fontsize=11)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("claim16_bpw.png", dpi=160, bbox_inches="tight")
print("Saved claim16_bpw.png")
