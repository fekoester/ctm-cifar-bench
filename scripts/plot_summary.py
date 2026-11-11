#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from ctm_cifar.plotting import _norm_loss_type, plot_summary


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def list_run_dirs(out_root: str) -> list[str]:
    out = []
    if not os.path.isdir(out_root):
        return out
    for name in sorted(os.listdir(out_root)):
        d = os.path.join(out_root, name)
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, "config.json")) and os.path.exists(os.path.join(d, "history.json")) and os.path.exists(os.path.join(d, "eval_per_tick.json")):
            out.append(d)
    return out


def make_label(cfg: dict[str, Any]) -> str:
    loss_type = _norm_loss_type(cfg.get("loss_type", "??"))
    zres = int(bool(cfg.get("z_residual", False)))
    lam = cfg.get("lambda_mono", None)
    dataset = cfg.get("dataset", "cifar?")
    T_ticks = cfg.get("T_ticks", None)

    if loss_type == "monotone":
        return f"{dataset} | mono | zres={zres} | λ={lam} | T={T_ticks}"
    else:
        return f"{dataset} | {loss_type} | zres={zres} | T={T_ticks}"




def read_run(run_dir: str) -> dict[str, Any]:
    cfg = load_json(os.path.join(run_dir, "config.json"))
    hist = load_json(os.path.join(run_dir, "history.json"))
    per_tick = load_json(os.path.join(run_dir, "eval_per_tick.json"))
    return {
        "run_dir": run_dir,
        "run_name": os.path.basename(run_dir),
        "cfg": cfg,
        "history": hist,
        "per_tick": per_tick,
        "label": make_label(cfg),
    }


def main():
    ap = argparse.ArgumentParser("Plot summary over run directories")
    ap.add_argument("--out_root", type=str, default="./runs")
    ap.add_argument("--dataset", type=str, default="", choices=["", "cifar10", "cifar100"])
    ap.add_argument("--keep_topk_cv", type=int, default=6)
    args = ap.parse_args()

    run_dirs = list_run_dirs(args.out_root)
    runs = [read_run(d) for d in run_dirs]

    if args.dataset.strip():
        want = args.dataset.strip().lower()
        runs = [r for r in runs if str(r["cfg"].get("dataset", "")).lower() == want]

    if not runs:
        print("No runs found/matched.")
        return

    out_dir = os.path.join(args.out_root, "summary_plots")
    plot_summary(runs, out_dir=out_dir, keep_topk_cv=args.keep_topk_cv)
    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()

