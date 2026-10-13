"""Plot effective bandwidth from gpu_results.json -> softmax_bandwidth.png"""
import json

import matplotlib.pyplot as plt

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# fixed categorical order; markers and direct labels carry identity, not color alone
SERIES = [
    ("triton", "Triton fused", "#2a78d6", "o"),
    ("torch_softmax", "torch.softmax", "#eb6834", "s"),
    ("torch_compile", "torch.compile", "#1baf7a", "^"),
    ("naive", "Unfused PyTorch", "#eda100", "D"),
]

meta = json.load(open("gpu_results.json"))
dtypes = list(dict.fromkeys(r["dtype"] for r in meta["results"]))
fig, axes = plt.subplots(1, len(dtypes), figsize=(5.2 * len(dtypes), 4.2), sharey=True, facecolor=SURFACE)
axes = axes if len(dtypes) > 1 else [axes]
ymax = max(max(r["gb_per_s"].values()) for r in meta["results"])
if meta.get("peak_gb_per_s"):
    ymax = max(ymax, meta["peak_gb_per_s"])
ymax *= 1.12

for ax, dtype in zip(axes, dtypes):
    rows = [r for r in meta["results"] if r["dtype"] == dtype]
    xs = [r["cols"] for r in rows]
    ax.set_facecolor(SURFACE)
    ends = []
    for key, label, color, marker in SERIES:
        ys = [r["gb_per_s"][key] for r in rows]
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=7, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ends.append([ys[-1], label])
    # direct labels at the right end, nudged apart so they never overlap
    ends.sort(key=lambda e: e[0])
    gap = ymax * 0.06
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, label in ends:
        ax.annotate(label, (xs[-1], y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    if meta.get("peak_gb_per_s"):
        ax.axhline(meta["peak_gb_per_s"], color=INK2, lw=1, ls="--", zorder=2)
        ax.text(xs[0], meta["peak_gb_per_s"], f" spec peak {meta['peak_gb_per_s']} GB/s",
                va="bottom", fontsize=8.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_ylim(0, ymax)
    ax.set_xlim(xs[0] / 1.3, xs[-1] * 2.6)
    ax.set_title(dtype, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Columns per row", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

axes[0].set_ylabel("Effective bandwidth (GB/s)", color=INK2)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper left", ncol=4,
           bbox_to_anchor=(0.005, 0.93), labelcolor=INK2)
fig.suptitle(f"Softmax over {meta['results'][0]['rows']} rows on {meta['gpu']}: higher is better",
             color=INK, fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig("softmax_bandwidth.png", dpi=160, facecolor=SURFACE)
print("Saved softmax_bandwidth.png")
