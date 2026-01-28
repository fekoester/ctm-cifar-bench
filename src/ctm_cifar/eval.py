from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import certainty_from_logits, per_tick_metrics_from_logits


@torch.no_grad()
def eval_cls(
    model,
    loader: DataLoader,
    device: str,
    eval_rule: str,
    additive_logits: bool,
    max_ticks: Optional[int],
) -> Dict[str, float]:
    model.eval()
    correct, total = 0, 0
    ce_sum = 0.0

    pbar = tqdm(loader, desc="eval", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits_all = model(x, max_ticks=max_ticks, additive_logits=additive_logits)  # [B,T,C]
        B, T, C = logits_all.shape

        if eval_rule == "final":
            logits = logits_all[:, -1, :]
        elif eval_rule == "certainty":
            cert = torch.stack([certainty_from_logits(logits_all[:, t, :]) for t in range(T)], dim=1)
            t_best = cert.argmax(dim=1)
            idx = torch.arange(B, device=device)
            logits = logits_all[idx, t_best, :]
        else:
            raise ValueError(f"bad eval_rule: {eval_rule}")

        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        ce_sum += F.cross_entropy(logits, y, reduction="sum").item()

        pbar.set_postfix(acc=f"{(correct/max(1,total))*100:.2f}%")

    return {"acc": correct / max(1, total), "ce": ce_sum / max(1, total)}


@torch.no_grad()
def eval_per_tick(
    model,
    loader: DataLoader,
    device: str,
    additive_logits: bool,
    max_ticks: Optional[int],
):
    model.eval()
    ce_sum = None
    acc_sum = None
    cert_sum = None
    n_total = 0

    pbar = tqdm(loader, desc="eval_per_tick", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits_all = model(x, max_ticks=max_ticks, additive_logits=additive_logits)  # [B,T,C]
        B, T, C = logits_all.shape

        ce_t, acc_t, cert_t = per_tick_metrics_from_logits(logits_all, y)  # [T]
        if ce_sum is None:
            ce_sum = torch.zeros(T, device=device)
            acc_sum = torch.zeros(T, device=device)
            cert_sum = torch.zeros(T, device=device)

        ce_sum += ce_t * B
        acc_sum += acc_t * B
        cert_sum += cert_t * B
        n_total += B

    ce_mean = (ce_sum / max(1, n_total)).detach().cpu().tolist()
    acc_mean = (acc_sum / max(1, n_total)).detach().cpu().tolist()
    cert_mean = (cert_sum / max(1, n_total)).detach().cpu().tolist()
    return {"ce": ce_mean, "acc": acc_mean, "cert": cert_mean}

