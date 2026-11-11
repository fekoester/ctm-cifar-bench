from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import certainty_from_logits, per_tick_metrics_from_logits

EVAL_RULES = ("final", "certainty", "best_tick", "mean_logits")


@torch.no_grad()
def collect_logits(
    model,
    loader: DataLoader,
    device: str,
    additive_logits: bool,
    max_ticks: int | None,
    desc: str = "collect",
):
    """Run the model once over a loader; return (logits [N,T,C] on CPU float32, labels [N])."""
    model.eval()
    outs, ys = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        logits_all = model(x, max_ticks=max_ticks, additive_logits=additive_logits)
        outs.append(logits_all.float().cpu())
        ys.append(y.cpu())
    return torch.cat(outs, 0), torch.cat(ys, 0)


def select_logits(logits_all: torch.Tensor, rule: str, tick: int | None = None) -> torch.Tensor:
    """
    Reduce [N,T,C] logits to one prediction per sample.
      final        last tick
      certainty    per-sample tick of maximal certainty (1 - normalised entropy)
      best_tick    a fixed tick index (choose it on the validation split, see choose_best_tick)
      mean_logits  average over ticks
    """
    N, T, _ = logits_all.shape
    if rule == "final":
        return logits_all[:, -1]
    if rule == "certainty":
        cert = certainty_from_logits(logits_all)  # [N,T]
        t_best = cert.argmax(dim=1)
        return logits_all[torch.arange(N), t_best]
    if rule == "best_tick":
        if tick is None:
            raise ValueError("best_tick needs a tick index")
        return logits_all[:, int(tick)]
    if rule == "mean_logits":
        return logits_all.mean(dim=1)
    raise ValueError(f"bad eval_rule: {rule} (choose from {EVAL_RULES})")


def choose_best_tick(logits_all: torch.Tensor, y: torch.Tensor) -> int:
    acc_t = (logits_all.argmax(-1) == y.unsqueeze(1)).float().mean(0)
    return int(acc_t.argmax().item())


def rule_metrics(logits_all: torch.Tensor, y: torch.Tensor, rule: str, tick: int | None = None) -> dict[str, float]:
    logits = select_logits(logits_all, rule, tick)
    return {
        "acc": (logits.argmax(-1) == y).float().mean().item(),
        "ce": F.cross_entropy(logits, y).item(),
    }


def per_tick_from_logits(logits_all: torch.Tensor, y: torch.Tensor) -> dict[str, list]:
    ce_t, acc_t, cert_t = per_tick_metrics_from_logits(logits_all, y)
    return {"ce": ce_t.tolist(), "acc": acc_t.tolist(), "cert": cert_t.tolist()}


def adaptive_halting_curve(
    logits_all: torch.Tensor, y: torch.Tensor, thresholds: Sequence[float]
) -> list:
    """
    Halt each sample at the first tick whose certainty reaches the threshold (else at the last tick)
    and predict from that tick. Returns accuracy vs mean ticks used, one point per threshold, plus
    the fixed-T points for comparison.
    """
    N, T, _ = logits_all.shape
    cert = certainty_from_logits(logits_all)  # [N,T]
    correct_t = logits_all.argmax(-1) == y.unsqueeze(1)  # [N,T]
    out = []
    for thr in thresholds:
        reached = cert >= thr
        reached[:, -1] = True
        t_stop = reached.float().argmax(dim=1)  # first True
        acc = correct_t[torch.arange(N), t_stop].float().mean().item()
        out.append({"threshold": float(thr), "acc": acc, "mean_ticks": (t_stop + 1).float().mean().item()})
    return out


@torch.no_grad()
def eval_cls(
    model,
    loader: DataLoader,
    device: str,
    eval_rule: str,
    additive_logits: bool,
    max_ticks: int | None,
    tick: int | None = None,
) -> dict[str, float]:
    logits_all, y = collect_logits(model, loader, device, additive_logits, max_ticks, desc="eval")
    return rule_metrics(logits_all, y, eval_rule, tick)


@torch.no_grad()
def eval_per_tick(
    model,
    loader: DataLoader,
    device: str,
    additive_logits: bool,
    max_ticks: int | None,
):
    logits_all, y = collect_logits(model, loader, device, additive_logits, max_ticks, desc="eval_per_tick")
    return per_tick_from_logits(logits_all, y)
