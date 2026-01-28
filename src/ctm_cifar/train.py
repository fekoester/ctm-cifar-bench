from __future__ import annotations

import os
import time
import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import TrainConfig
from .data import build_cifar_loaders
from .model import CTMCIFARClassifier, CTMOptions
from .losses import loss_minmax_cls, loss_monotone_cls, loss_allticks_monotone_cls
from .eval import eval_per_tick
from .logging_utils import safe_makedirs, write_json, append_jsonl, CSVLogger, now_unix
from .metrics import per_tick_metrics_from_logits


CSV_FIELDS = [
    "run_id", "timestamp", "split",
    "epoch", "step", "samples_seen",
    "loss_total", "aux1", "aux2",
    "lr", "sec_per_step", "samples_per_sec",
    "val_acc_2T", "val_ce_2T",
]


class WarmupCosine:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, base_lr: float):
        self.opt = optimizer
        self.warmup_steps = int(max(0, warmup_steps))
        self.total_steps = int(max(1, total_steps))
        self.base_lr = float(base_lr)
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        s = self.step_num
        if self.warmup_steps > 0 and s <= self.warmup_steps:
            lr = self.base_lr * (s / self.warmup_steps)
        else:
            t = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            t = min(max(t, 0.0), 1.0)
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * t))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


def _set_seeds(seed: int):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def make_run_dir(cfg: TrainConfig) -> str:
    safe_makedirs(cfg.out_root)
    run_id = cfg.stable_run_id()
    tag = cfg.run_name.strip()
    name = f"{tag}_" if tag else ""
    # include dataset/loss for human readability
    human = f"{cfg.dataset}_loss={cfg.loss_type}_zres={int(cfg.z_residual)}_T={cfg.T_train}"
    run_dir = os.path.join(cfg.out_root, f"{name}{human}_{run_id}")
    safe_makedirs(run_dir)
    return run_dir


def train_one(cfg: TrainConfig) -> str:
    device = _resolve_device(cfg.device)
    cfg.device = device

    run_dir = make_run_dir(cfg)
    write_json(os.path.join(run_dir, "config.json"), cfg.to_dict())

    csv_logger = CSVLogger(os.path.join(cfg.out_root, "runs.csv"), CSV_FIELDS)
    jsonl_path = os.path.join(run_dir, "events.jsonl")

    _set_seeds(cfg.seed)

    train_loader, val_loader, num_classes = build_cifar_loaders(
        dataset=cfg.dataset,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    opt = CTMOptions(
        D=cfg.D, M=cfg.M, H=cfg.H, T_ticks=cfg.T_eval,
        feat_dim=cfg.feat_dim, attn_dim=cfg.attn_dim, attn_heads=cfg.attn_heads,
        Dout_pairs=cfg.Dout_pairs, Dact_pairs=cfg.Dact_pairs,
        self_pairs_out=cfg.self_pairs_out, self_pairs_act=cfg.self_pairs_act,
        pair_seed=cfg.pair_seed,
        syn_depth=cfg.syn_depth, syn_bottleneck=cfg.syn_bottleneck,
        dropout=cfg.dropout,
        z_residual=cfg.z_residual,
        eta=cfg.eta,
        layernorm_z=cfg.layernorm_z,
        layernorm_o=cfg.layernorm_o,
        beta_per_sample=cfg.beta_per_sample,
        state_feat=cfg.state_feat,
        hyper_rank=cfg.hyper_rank,
        hyper_hidden=cfg.hyper_hidden,
    )
    model = CTMCIFARClassifier(num_classes=num_classes, opt=opt).to(device)

    use_amp = bool(cfg.amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = cfg.steps_per_epoch if cfg.steps_per_epoch > 0 else len(train_loader)
    total_steps = int(cfg.epochs) * int(steps_per_epoch)
    sched = WarmupCosine(optimizer, warmup_steps=cfg.warmup_steps, total_steps=total_steps, base_lr=cfg.lr)

    history = []
    track = {
        "steps": [],
        "loss": [],
        "acc": [],
        "cert": [],
        "aux1": [],
        "aux2": [],
        "total": [],
        "meta": {},
    }

    if cfg.loss_type == "minmax":
        track["meta"] = {"aux1_name": "minCE", "aux2_name": "maxCertCE"}
    elif cfg.loss_type == "monotone":
        track["meta"] = {"aux1_name": "finalCE", "aux2_name": "mono_pen"}
    elif cfg.loss_type == "allticks":
        track["meta"] = {"aux1_name": "meanCE_allticks", "aux2_name": "mono_pen"}
    else:
        track["meta"] = {"aux1_name": "CE(stop)", "aux2_name": "stop_tick"}

    run_id = cfg.stable_run_id()
    global_step = 0
    samples_seen = 0
    grad_accum = max(1, int(cfg.grad_accum))

    optimizer.zero_grad(set_to_none=True)
    t_step_last = time.time()

    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        ep_loss_sum = 0.0
        ep_seen = 0
        ep_start = time.time()

        pbar = tqdm(train_loader, desc=f"train ep{ep}", leave=True)
        for step_in_epoch, (x, y) in enumerate(pbar, start=1):
            global_step += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if cfg.loss_type == "stochastic":
                    N = int(torch.randint(low=1, high=int(cfg.T_train) + 1, size=(1,)).item())
                    if cfg.stoch_truncate_compute:
                        logits_all = model(x, max_ticks=N, additive_logits=cfg.additive_logits)
                        logits_stop = logits_all[:, -1, :]
                    else:
                        logits_all = model(x, max_ticks=cfg.T_train, additive_logits=cfg.additive_logits)
                        logits_stop = logits_all[:, N - 1, :]
                    loss = F.cross_entropy(logits_stop, y)
                    aux1 = loss.detach()
                    aux2 = torch.tensor(float(N), device=loss.device)
                else:
                    logits_all = model(x, max_ticks=cfg.T_train, additive_logits=cfg.additive_logits)
                    if cfg.loss_type == "monotone":
                        loss, aux1, aux2 = loss_monotone_cls(
                            logits_all, y, lambda_mono=cfg.lambda_mono, margin=cfg.mono_margin
                        )
                    elif cfg.loss_type == "allticks":
                        loss, aux1, aux2 = loss_allticks_monotone_cls(
                            logits_all, y, lambda_mono=cfg.lambda_mono, margin=cfg.mono_margin
                        )
                    else:
                        loss, aux1, aux2 = loss_minmax_cls(logits_all, y)

                loss_scaled = loss / grad_accum

            scaler.scale(loss_scaled).backward()

            bs = y.size(0)
            ep_seen += bs
            samples_seen += int(bs)
            ep_loss_sum += float(loss.detach().cpu()) * bs

            # timing
            t_now = time.time()
            sec_per_step = t_now - t_step_last
            t_step_last = t_now
            samples_per_sec = bs / max(1e-9, sec_per_step)

            do_step = (global_step % grad_accum) == 0
            if do_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_now = sched.step()
            else:
                grad_norm = None
                lr_now = optimizer.param_groups[0]["lr"]

            avg_loss = ep_loss_sum / max(1, ep_seen)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", aux1=f"{float(aux1):.3f}", aux2=f"{float(aux2):.3f}", lr=f"{lr_now:.2e}")

            # per-tick tracking snapshot
            if (global_step % int(cfg.track_every)) == 0:
                model.eval()
                with torch.no_grad():
                    logits_full = model(x, max_ticks=cfg.T_train, additive_logits=cfg.additive_logits).detach()
                    loss_t, acc_t, cert_t = per_tick_metrics_from_logits(logits_full, y)
                model.train()

                track["steps"].append(global_step)
                track["loss"].append(loss_t.cpu().tolist())
                track["acc"].append(acc_t.cpu().tolist())
                track["cert"].append(cert_t.cpu().tolist())
                track["aux1"].append(float(aux1.detach().cpu()))
                track["aux2"].append(float(aux2.detach().cpu()))
                track["total"].append(float(loss.detach().cpu()))

            # global logs
            if (global_step % int(cfg.log_every)) == 0:
                row = {
                    "run_id": run_id,
                    "timestamp": now_unix(),
                    "split": "train",
                    "epoch": ep,
                    "step": global_step,
                    "samples_seen": samples_seen,
                    "loss_total": float(loss.detach().cpu()),
                    "aux1": float(aux1.detach().cpu()),
                    "aux2": float(aux2.detach().cpu()),
                    "lr": float(lr_now),
                    "sec_per_step": float(sec_per_step),
                    "samples_per_sec": float(samples_per_sec),
                    "val_acc_2T": "",
                    "val_ce_2T": "",
                }
                csv_logger.log(row)
                append_jsonl(jsonl_path, row)

            if cfg.steps_per_epoch > 0 and step_in_epoch >= cfg.steps_per_epoch:
                break

        ep_time = time.time() - ep_start
        train_loss = ep_loss_sum / max(1, ep_seen)

        # end-of-epoch: eval per tick at 2T
        per_tick_2T = eval_per_tick(
            model, val_loader, device=device, additive_logits=cfg.additive_logits, max_ticks=cfg.T_eval
        )
        val_acc_2T = float(per_tick_2T["acc"][-1])
        val_ce_2T = float(per_tick_2T["ce"][-1])

        hist_row = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "time_sec": float(ep_time),
            "throughput_samp_per_sec": float(ep_seen / max(1e-9, ep_time)),
            "val_acc_2T": val_acc_2T,
            "val_ce_2T": val_ce_2T,
            "val_acc_ticks_2T": per_tick_2T["acc"],
            "val_ce_ticks_2T": per_tick_2T["ce"],
            "val_cert_ticks_2T": per_tick_2T["cert"],
        }
        history.append(hist_row)

        # log val row to CSV/jsonl
        val_row = {
            "run_id": run_id,
            "timestamp": now_unix(),
            "split": "val",
            "epoch": ep,
            "step": global_step,
            "samples_seen": samples_seen,
            "loss_total": "",
            "aux1": "",
            "aux2": "",
            "lr": "",
            "sec_per_step": "",
            "samples_per_sec": "",
            "val_acc_2T": val_acc_2T,
            "val_ce_2T": val_ce_2T,
        }
        csv_logger.log(val_row)
        append_jsonl(jsonl_path, val_row)

        # write checkpoints/logs
        write_json(os.path.join(run_dir, "history.json"), history)
        torch.save(track, os.path.join(run_dir, "tick_tracking.pt"))
        write_json(os.path.join(run_dir, "eval_per_tick.json"), {
            "acc": per_tick_2T["acc"],
            "ce": per_tick_2T["ce"],
            "cert": per_tick_2T["cert"],
        })

        print(
            f"epoch {ep:03d} | train_loss={train_loss:.4f} | "
            f"val_ce_2T={val_ce_2T:.4f} | val_acc_2T={val_acc_2T*100:.2f}% | "
            f"time={ep_time:.1f}s | {hist_row['throughput_samp_per_sec']:.0f} samp/s"
        )

    csv_logger.close()
    return run_dir

