import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

assets    = ["CPER","LIT","PPLT","PALL","SLV","GLD","JJN","JJU"]
strategy  = [0.43,  0.72, 0.716,  0.82,  0.89, 1.19, 0.71, 1.14]
benchmark = [0.49, 0.178,0.62,-0.005, 0.838,1.087,0.419,0.217]

x   = np.arange(len(assets))
w   = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#FAFAF8")
ax.set_facecolor("#FAFAF8")

bars1 = ax.bar(x - w/2, strategy,  width=w, color="#6D28D9", label="Strategy (best params)")
bars2 = ax.bar(x + w/2, benchmark, width=w, color="#C4B5FD", label="Benchmark (buy & hold)")

# Value labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
            f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, color="#1A1523")

for bar in bars2:
    h = bar.get_height()
    offset = 0.02 if h >= 0 else -0.06
    ax.text(bar.get_x() + bar.get_width()/2, h + offset,
            f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, color="#71717A")

ax.set_xticks(x)
ax.set_xticklabels(assets, fontsize=11, color="#1A1523")
ax.set_ylim(-0.25, 1.45)
ax.axhline(0, color="#E4E0F0", linewidth=0.8)
ax.yaxis.grid(True, color="#EDE9FE", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(left=False, colors="#71717A")

ax.set_title("Strategy (best params) vs. buy-and-hold benchmark — Sharpe ratio",
             fontsize=12, color="#1A1523", pad=14, loc="left")

ax.legend(frameon=False, fontsize=10, labelcolor="#71717A")

plt.tight_layout()
plt.savefig("sharpe_chart.png", dpi=180, bbox_inches="tight", facecolor="#FAFAF8")
plt.show()