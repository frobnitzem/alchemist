"""Usage: python runs/plot.py runs/<name>   (reads output.safetensors, writes figures/*.png)"""
import sys, pathlib
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors.torch import load_file

run = pathlib.Path(sys.argv[1]); out = run / "figures"; out.mkdir(exist_ok=True)
d = {k: v.numpy() for k, v in load_file(run / "output.safetensors").items()}
pb, eb = d["site.pbins"], d["site.ebins"]
names = ["center A", "center B"]

for key, ybins, ylabel, fn in [("site.H_prob1_prob2", pb, "prob2 (NN2 A-fraction)", "prob1_vs_prob2.png"),
                               ("site.H_prob1_e1", eb, "NN1 energy contribution", "prob1_vs_e1.png")]:
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for t in range(2):
        H = d[key][t]
        im = ax[t].pcolormesh(pb, ybins, np.log10(H.T + 1), cmap="viridis")
        ax[t].set(title=names[t], xlabel="prob1 (NN1 A-fraction)")
        fig.colorbar(im, ax=ax[t], label="log10(count+1)")
    ax[0].set_ylabel(ylabel); fig.tight_layout(); fig.savefig(out / fn, dpi=130); plt.close(fig)

m, w = d["site.site_mean"], d["site.site_weight"]
c = d["composition.composition"][:, 0]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
x = np.arange(4); lab = ["prob1", "prob2", "E1", "E2"]
ax[0].bar(x - .2, m[0], .4, label="center A"); ax[0].bar(x + .2, m[1], .4, label="center B")
ax[0].set_xticks(x, lab); ax[0].legend(); ax[0].set_title("per-site means by type")
ax[1].plot(c[1:5], "o", label="NN1"); ax[1].plot(range(4, 16), c[5:17], "s", label="NN2")
ax[1].axhline(c[0], ls="--", c="gray", label="P(A)")
ax[1].set(title="P(neighbor A | center A) by slot", xlabel="slot", ylim=(0, 1)); ax[1].legend()
fig.tight_layout(); fig.savefig(out / "site_summary.png", dpi=130)
print(f"P(A)={c[0]:.3f} weights A,B={w}\nmeans [p1,p2,E1,E2]\n A: {m[0]}\n B: {m[1]}")
