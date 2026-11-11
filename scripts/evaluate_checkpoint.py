#!/usr/bin/env python3
"""
Evaluate a saved checkpoint (best.pt / last.pt) on the test split: every eval rule, per-tick
accuracy, and the adaptive-halting curve (accuracy vs mean ticks for each certainty threshold).

    python scripts/evaluate_checkpoint.py runs/<run>/best.pt --T_eval 40 --out eval.json
"""
from __future__ import annotations

import argparse
import json

import torch

from ctm_cifar.data import build_cifar_loaders
from ctm_cifar.train import evaluate_split, load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--T_eval", type=int, default=None, help="override the number of ticks")
    ap.add_argument("--out", default=None, help="write the metrics as JSON")
    args = ap.parse_args()

    model, cfg, ckpt = load_checkpoint(args.checkpoint, args.device)
    if args.T_eval:
        cfg.T_eval_mult = 1
        cfg.T_train = args.T_eval
        if hasattr(model, "T_ticks"):
            model.T_ticks = max(int(model.T_ticks), args.T_eval)
    _, _, test_loader, _ = build_cifar_loaders(
        cfg.dataset, args.data_dir or cfg.data_dir, args.batch_size, 2, cfg.seed, cfg.val_size
    )
    res = evaluate_split(model, test_loader, cfg, args.device, best_tick=ckpt.get("best_tick"),
                         halt_thresholds=list(cfg.halt_thresholds))
    print(f"checkpoint epoch {ckpt['epoch']} | {cfg.T_eval} ticks")
    for r, m in res["rules"].items():
        print(f"  {r:12s} acc={m['acc']*100:6.2f}%  ce={m['ce']:.4f}")
    print("  adaptive halting (threshold -> acc @ mean ticks):")
    for p in res.get("adaptive_halting", []):
        print(f"    {p['threshold']:.2f} -> {p['acc']*100:6.2f}% @ {p['mean_ticks']:.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
