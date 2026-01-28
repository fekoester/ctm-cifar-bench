#!/usr/bin/env python3
from __future__ import annotations

import argparse
from ctm_cifar.config import TrainConfig
from ctm_cifar.train import train_one


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("CTM CIFAR training")

    # system
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--amp", type=int, default=1, choices=[0, 1])
    ap.add_argument("--seed", type=int, default=0)

    # data
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--data_dir", type=str, default="./data_cifar")
    ap.add_argument("--num_workers", type=int, default=4)

    # training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--steps_per_epoch", type=int, default=0)

    # model size
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--T_train", type=int, default=10)
    ap.add_argument("--T_eval_mult", type=int, default=2)

    ap.add_argument("--feat_dim", type=int, default=64)
    ap.add_argument("--attn_dim", type=int, default=64)
    ap.add_argument("--attn_heads", type=int, default=16)

    ap.add_argument("--Dout_pairs", type=int, default=512)
    ap.add_argument("--Dact_pairs", type=int, default=512)
    ap.add_argument("--self_pairs_out", type=int, default=0)
    ap.add_argument("--self_pairs_act", type=int, default=0)
    ap.add_argument("--pair_seed", type=int, default=123)

    ap.add_argument("--syn_depth", type=int, default=10)
    ap.add_argument("--syn_bottleneck", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)

    # knobs
    ap.add_argument("--z_residual", type=int, default=1, choices=[0, 1])
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--layernorm_z", type=int, default=1, choices=[0, 1])
    ap.add_argument("--layernorm_o", type=int, default=1, choices=[0, 1])
    ap.add_argument("--beta_per_sample", type=int, default=0, choices=[0, 1])

    # feature mode
    ap.add_argument("--state_feat", type=int, default=0, choices=[0, 1])
    ap.add_argument("--hyper_rank", type=int, default=16)
    ap.add_argument("--hyper_hidden", type=int, default=128)

    # logits refinement
    ap.add_argument("--additive_logits", type=int, default=0, choices=[0, 1])

    # objectives
    ap.add_argument("--loss_type", type=str, default="minmax",
                    choices=["minmax", "monotone", "allticks", "stochastic"])
    ap.add_argument("--eval_rule", type=str, default="certainty", choices=["final", "certainty"])
    ap.add_argument("--lambda_mono", type=float, default=0.05)
    ap.add_argument("--mono_margin", type=float, default=0.0)
    ap.add_argument("--stoch_truncate_compute", type=int, default=0, choices=[0, 1])

    # logging
    ap.add_argument("--out_root", type=str, default="./runs")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--track_every", type=int, default=50)
    ap.add_argument("--log_every", type=int, default=50)

    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()

    cfg = TrainConfig(
        device=args.device,
        amp=bool(args.amp),
        seed=args.seed,

        dataset=args.dataset,
        data_dir=args.data_dir,
        num_workers=args.num_workers,

        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        steps_per_epoch=args.steps_per_epoch,

        D=args.D,
        M=args.M,
        H=args.H,
        T_train=args.T_train,
        T_eval_mult=args.T_eval_mult,

        feat_dim=args.feat_dim,
        attn_dim=args.attn_dim,
        attn_heads=args.attn_heads,

        Dout_pairs=args.Dout_pairs,
        Dact_pairs=args.Dact_pairs,
        self_pairs_out=args.self_pairs_out,
        self_pairs_act=args.self_pairs_act,
        pair_seed=args.pair_seed,

        syn_depth=args.syn_depth,
        syn_bottleneck=args.syn_bottleneck,
        dropout=args.dropout,

        z_residual=bool(args.z_residual),
        eta=args.eta,
        layernorm_z=bool(args.layernorm_z),
        layernorm_o=bool(args.layernorm_o),
        beta_per_sample=bool(args.beta_per_sample),

        state_feat=bool(args.state_feat),
        hyper_rank=args.hyper_rank,
        hyper_hidden=args.hyper_hidden,

        additive_logits=bool(args.additive_logits),

        loss_type=args.loss_type,
        eval_rule=args.eval_rule,
        lambda_mono=args.lambda_mono,
        mono_margin=args.mono_margin,
        stoch_truncate_compute=bool(args.stoch_truncate_compute),

        out_root=args.out_root,
        run_name=args.run_name,
        track_every=args.track_every,
        log_every=args.log_every,
    )

    run_dir = train_one(cfg)
    print("\nSaved run to:", run_dir)


if __name__ == "__main__":
    main()

