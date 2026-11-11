#!/usr/bin/env python3
"""
Compare finished runs from their final_metrics.json: test accuracy per tick (left) and the
adaptive-halting trade-off, accuracy vs mean ticks used (right).

    python scripts/plot_compare.py runs/legacy_* runs/fixed_* --labels legacy fixed --out compare.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="compare.png")
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(d.rstrip("/")) for d in args.run_dirs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for d, lab in zip(args.run_dirs, labels):
        fm = json.load(open(os.path.join(d, "final_metrics.json")))
        test = fm["test"]
        acc = [100 * a for a in test["per_tick"]["acc"]]
        if len(acc) == 1:  # single-tick baseline: a horizontal reference line
            ax1.axhline(acc[0], ls="--", color="gray", label=f"{lab} ({acc[0]:.1f}%)")
            ax2.axhline(acc[0], ls="--", color="gray", label=lab)
            continue
        (line,) = ax1.plot(range(1, len(acc) + 1), acc, marker=".", label=lab)
        curve = test.get("adaptive_halting", [])
        if curve:
            ax2.plot([p["mean_ticks"] for p in curve], [100 * p["acc"] for p in curve], marker="o",
                     color=line.get_color(), label=f"{lab}: certainty halting")
        ax2.plot(range(1, len(acc) + 1), acc, ls=":", color=line.get_color(), alpha=0.7,
                 label=f"{lab}: fixed T")
    ax1.set_xlabel("tick")
    ax1.set_ylabel("test accuracy (%)")
    ax1.set_title("Accuracy per tick")
    ax2.set_xlabel("mean ticks used")
    ax2.set_ylabel("test accuracy (%)")
    ax2.set_title("Adaptive compute: halt when certainty ≥ threshold")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
