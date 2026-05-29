"""Read benchmark.json and produce box-and-whisker plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("time_to_goal",    "Time to goal [s]"),
    ("path_length",     "Path length [m]"),
    ("mean_lin_speed",  "|v| mean [m/s]"),
    ("mean_ang_speed",  "|ω| mean [rad/s]"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/benchmark.json")
    ap.add_argument("--out", default="results/boxplots.png")
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text())
    names = list(data.keys())
    pretty = [data[k]["name"] for k in names]
    colors_rgb = [tuple(c / 255 for c in data[k]["color"]) for k in names]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor("#f7f5ef")

    for ax, (key, label) in zip(axes.flat, METRICS):
        series = [[r[key] for r in data[k]["rows"]] for k in names]
        bp = ax.boxplot(
            series, labels=pretty, patch_artist=True,
            medianprops=dict(color="#222", linewidth=1.6),
            whiskerprops=dict(color="#555"),
            capprops=dict(color="#555"),
            flierprops=dict(marker="o", markerfacecolor="#444",
                            markeredgecolor="none", markersize=3, alpha=0.7),
        )
        for patch, c in zip(bp["boxes"], colors_rgb):
            patch.set_facecolor(c + (0.55,)) if len(c) == 3 else patch.set_facecolor(c)
            patch.set_edgecolor("#222")
            patch.set_linewidth(1.2)
        ax.set_title(label, loc="left", fontsize=11, fontweight="600",
                     color="#222", pad=10)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")
        ax.set_facecolor("#fdfcf8")
        ax.tick_params(axis="x", labelsize=9)

    # Suptitle with success rates
    rates = ", ".join(
        f"{data[k]['name']}: {data[k]['success_rate']*100:.0f}%"
        for k in names
    )
    fig.suptitle("Bicycle stabilization to (0, 0, 0) — controller comparison\n"
                 f"goal-reach rate (pos < 0.3 m): {rates}",
                 fontsize=12, color="#111", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
