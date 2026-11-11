#!/usr/bin/env python3
"""
Train CTM (or the CNN baseline) on CIFAR.

Every TrainConfig field is a flag with the same name. Boolean flags take 0/1 and may be given bare
(`--amp` == `--amp 1`). Sweeps:

  --sweep4                      loss {minmax, monotone} x z_residual {0, 1}
  --lambda_sweep 0,0.05,0.1     monotone runs with each lambda_mono (combined with --sweep4: the
                                monotone half of the sweep is repeated for each lambda)
  --seeds 0,1,2                 repeat everything over seeds
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
from typing import get_type_hints

from ctm_cifar.config import TrainConfig
from ctm_cifar.train import train_one

CHOICES = {
    "device": ["cpu", "cuda"],
    "dataset": ["cifar10", "cifar100", "fake"],
    "model_type": ["ctm", "cnn"],
    "loss_type": ["minmax", "monotone", "allticks", "stochastic"],
    "eval_rule": ["final", "certainty", "best_tick", "mean_logits"],
}


def _bool01(v: str) -> bool:
    if str(v).lower() in ("1", "true", "yes"):
        return True
    if str(v).lower() in ("0", "false", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected 0/1, got {v!r}")


def _floats(v: str) -> list[float]:
    return [float(t) for t in v.split(",") if t.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("CTM CIFAR training")
    hints = get_type_hints(TrainConfig)
    for f in dataclasses.fields(TrainConfig):
        default = f.default
        typ = hints[f.name]
        if typ is bool:
            ap.add_argument(f"--{f.name}", type=_bool01, nargs="?", const=True, default=default)
        elif f.name == "halt_thresholds":
            ap.add_argument("--halt_thresholds", type=lambda v: tuple(_floats(v)), default=default)
        elif typ in (int, float, str):
            ap.add_argument(f"--{f.name}", type=typ, default=default, choices=CHOICES.get(f.name))
    ap.add_argument("--sweep4", action="store_true", help="minmax/monotone x z_residual 0/1")
    ap.add_argument("--lambda_sweep", type=_floats, default=None, help="comma-separated lambda_mono values")
    ap.add_argument("--seeds", type=lambda v: [int(t) for t in v.split(",")], default=None)
    return ap


def expand_runs(args: argparse.Namespace) -> list[TrainConfig]:
    names = {f.name for f in dataclasses.fields(TrainConfig)}
    base = {k: v for k, v in vars(args).items() if k in names}
    variants: list[dict] = []
    lambdas = args.lambda_sweep or [base["lambda_mono"]]
    if args.sweep4:
        for zres in (False, True):
            variants.append({"loss_type": "minmax", "z_residual": zres})
            variants += [{"loss_type": "monotone", "z_residual": zres, "lambda_mono": lam} for lam in lambdas]
    elif args.lambda_sweep:
        variants = [{"loss_type": "monotone", "lambda_mono": lam} for lam in lambdas]
    else:
        variants = [{}]
    seeds = args.seeds or [base["seed"]]
    return [TrainConfig(**{**base, **v, "seed": s}) for v, s in itertools.product(variants, seeds)]


def main():
    args = build_parser().parse_args()
    runs = expand_runs(args)
    for i, cfg in enumerate(runs, 1):
        if len(runs) > 1:
            print(f"\n=== run {i}/{len(runs)}: loss={cfg.loss_type} zres={int(cfg.z_residual)} "
                  f"lambda={cfg.lambda_mono} seed={cfg.seed} ===")
        run_dir = train_one(cfg)
        print("Saved run to:", run_dir)


if __name__ == "__main__":
    main()
