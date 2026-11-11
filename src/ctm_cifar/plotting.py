from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .logging_utils import safe_makedirs

BASE_COLOR = {
    "monotone":   "C0",
    "minmax":     "C1",
    "allticks":   "C2",
    "stochastic": "C3",
}
LSTYLES = ["-", "--", ":", "-."]


def _norm_loss_type(s: str) -> str:
    s = str(s).strip().lower()
    if s.startswith("mono"):
        return "monotone"
    if s.startswith("min"):
        return "minmax"
    return s


def run_style_key(cfg: dict[str, Any]) -> tuple:
    dataset = str(cfg.get("dataset", "cifar?")).lower()
    loss_type = _norm_loss_type(cfg.get("loss_type", "??"))
    zres = int(bool(cfg.get("z_residual", False)))
    lam = cfg.get("lambda_mono", None)
    lam_f = float(lam) if lam is not None else -1.0
    return (dataset, loss_type, zres, lam_f)



def _lambda_groups_for_monotone(runs: list[dict[str, Any]]) -> dict[tuple, int]:
    lambdas_by_group = {}
    for r in runs:
        cfg = r["cfg"]
        dataset = str(cfg.get("dataset", "cifar?")).lower()
        loss_type = _norm_loss_type(cfg.get("loss_type", "??"))
        zres = int(bool(cfg.get("z_residual", False)))
        if loss_type in ("monotone", "allticks"):
            key = (dataset, loss_type, zres)
            lam = cfg.get("lambda_mono", None)
            lam_f = float(lam) if lam is not None else -1.0
            lambdas_by_group.setdefault(key, set()).add(lam_f)

    idx_map = {}
    for key, lam_set in lambdas_by_group.items():
        for i, lam in enumerate(sorted(list(lam_set))):
            idx_map[(key, lam)] = i
    return idx_map


def _style_for_run(cfg: dict[str, Any], lam_idx_map: dict[tuple, int]) -> dict[str, Any]:
    dataset = str(cfg.get("dataset", "cifar?")).lower()
    loss_type = _norm_loss_type(cfg.get("loss_type", "??"))
    zres = int(bool(cfg.get("z_residual", False)))
    lam = cfg.get("lambda_mono", None)
    lam_f = float(lam) if lam is not None else -1.0

    color = BASE_COLOR.get(loss_type, "C7")

    # Requested: ONLY encode zres with linestyle
    ls = "-" if zres == 0 else "--"

    # Optional: if you sweep lambda, you can still vary linewidth a tiny bit
    # without breaking the "only two linestyles" rule:
    lw = 2.2
    if loss_type in ("monotone", "allticks"):
        lam_index = lam_idx_map.get(((dataset, loss_type, zres), lam_f), 0)
        lw = 2.0 + 0.3 * (lam_index % 3)

    return {"color": color, "linestyle": ls, "linewidth": lw}



def extract_history_arrays(history: list[dict[str, Any]]):
    epochs = np.array([h.get("epoch", i + 1) for i, h in enumerate(history)], dtype=np.int64)
    val_acc = np.array([h.get("val_acc_2T", np.nan) for h in history], dtype=np.float32)
    val_ce = np.array([h.get("val_ce_2T", np.nan) for h in history], dtype=np.float32)
    train_loss = np.array([h.get("train_loss", np.nan) for h in history], dtype=np.float32)
    return epochs, train_loss, val_acc, val_ce


def extract_best_tick_acc_over_epochs(history: list[dict[str, Any]], mode: str = "max_acc"):
    epochs = np.array([h.get("epoch", i + 1) for i, h in enumerate(history)], dtype=np.int64)
    best = []
    for h in history:
        acc = np.asarray(h.get("val_acc_ticks_2T", []), dtype=np.float32)
        cert = np.asarray(h.get("val_cert_ticks_2T", []), dtype=np.float32)
        if acc.size == 0:
            best.append(np.nan)
            continue
        if mode == "max_acc":
            best.append(float(acc.max()))
        elif mode == "cert_tick_acc":
            if cert.size == 0:
                best.append(float(acc.max()))
            else:
                best.append(float(acc[int(cert.argmax())]))
        else:
            raise ValueError(mode)
    return epochs, np.asarray(best, dtype=np.float32)


def plot_summary(runs: list[dict[str, Any]], out_dir: str, keep_topk_cv: int = 6):
    safe_makedirs(out_dir)
    lam_idx_map = _lambda_groups_for_monotone(runs)

    # val acc vs epoch
    plt.figure(figsize=(6, 4))
    for r in sorted(runs, key=lambda x: run_style_key(x["cfg"])):
        epochs, _, val_acc, _ = extract_history_arrays(r["history"])
        st = _style_for_run(r["cfg"], lam_idx_map)
        plt.plot(epochs, val_acc * 100.0, label=r["label"], **st)
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy @2T final tick (%)")
    plt.title("Validation accuracy vs epoch (overlay)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_acc_vs_epoch.png"), dpi=220)
    plt.close()

    # val ce vs epoch
    plt.figure(figsize=(6, 4))
    for r in sorted(runs, key=lambda x: run_style_key(x["cfg"])):
        epochs, _, _, val_ce = extract_history_arrays(r["history"])
        st = _style_for_run(r["cfg"], lam_idx_map)
        plt.plot(epochs, val_ce, label=r["label"], **st)
    plt.xlabel("epoch")
    plt.ylabel("validation cross-entropy @2T final tick")
    plt.title("Validation cross-entropy vs epoch (overlay)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_ce_vs_epoch.png"), dpi=220)
    plt.close()

    # best tick curves
    for mode, fname, ttl in [
        ("max_acc", "val_acc_besttick_vs_epoch.png", "Best tick acc per epoch [argmax acc over ticks]"),
        ("cert_tick_acc", "val_acc_certtick_vs_epoch.png", "Best tick acc per epoch [tick = argmax certainty]"),
    ]:
        plt.figure(figsize=(6, 4))
        for r in sorted(runs, key=lambda x: run_style_key(x["cfg"])):
            epochs, best = extract_best_tick_acc_over_epochs(r["history"], mode=mode)
            st = _style_for_run(r["cfg"], lam_idx_map)
            plt.plot(epochs, best * 100.0, label=r["label"], **st)
        plt.xlabel("epoch")
        plt.ylabel("validation accuracy (%)")
        plt.title(ttl)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=220)
        plt.close()

    # end-of-run per tick (from eval_per_tick.json)
    def _plot_per_tick(key: str, ylabel: str, fname: str, title: str, scale100: bool = False):
        plt.figure(figsize=(6, 4))
        for r in sorted(runs, key=lambda x: run_style_key(x["cfg"])):
            arr = np.asarray(r["per_tick"].get(key, []), dtype=np.float32)
            if arr.size == 0:
                continue
            ticks = np.arange(1, arr.size + 1)
            st = _style_for_run(r["cfg"], lam_idx_map)
            y = arr * 100.0 if scale100 else arr
            plt.plot(ticks, y, label=r["label"], **st)
        plt.xlabel("tick")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=220)
        plt.close()

    _plot_per_tick("acc", "accuracy (%)", "eval_per_tick_acc.png", "End-of-run per-tick accuracy", scale100=True)
    _plot_per_tick("ce", "cross-entropy", "eval_per_tick_ce.png", "End-of-run per-tick cross-entropy")
    _plot_per_tick("cert", "certainty (1 - H/logC)", "eval_per_tick_cert.png", "End-of-run per-tick certainty")

    # CV figure: keep top-k by final val_acc_2T
    runs_sorted = sorted(
        runs,
        key=lambda r: float(r["history"][-1].get("val_acc_2T", -1.0)) if r["history"] else -1.0,
        reverse=True,
    )
    keep = runs_sorted[: max(1, int(keep_topk_cv))]
    keep_sorted = sorted(keep, key=lambda x: run_style_key(x["cfg"]))

    plt.figure(figsize=(10, 5.8))
    for r in keep_sorted:
        acc = np.asarray(r["per_tick"].get("acc", []), dtype=np.float32)
        if acc.size == 0:
            continue
        ticks = np.arange(1, acc.size + 1)
        cfg = r["cfg"]
        loss_type = _norm_loss_type(cfg.get("loss_type", "??"))
        zres = int(bool(cfg.get("z_residual", False)))
        lam = cfg.get("lambda_mono", None)

        if loss_type in ("monotone", "allticks"):
            lbl = f"{loss_type} (zres={zres}, λ={lam})"
        else:
            lbl = f"{loss_type} (zres={zres})"

        st = _style_for_run(cfg, lam_idx_map)
        plt.plot(ticks, acc * 100.0, label=lbl, **st)

    plt.xlabel("tick")
    plt.ylabel("validation accuracy (%)")
    plt.title("CTM CIFAR — per-tick validation accuracy (CV figure)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_figure.png"), dpi=260)
    plt.close()

