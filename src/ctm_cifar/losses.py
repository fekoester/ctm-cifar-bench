from __future__ import annotations

import torch
import torch.nn.functional as F
from .metrics import certainty_from_logits


def loss_minmax_cls(logits_all: torch.Tensor, y: torch.Tensor):
    """
    total = 0.5*(min_t CE_t + CE_{t_max_certainty})
    returns (total, aux1=minCE, aux2=maxCertCE)
    """
    B, T, C = logits_all.shape
    ce = []
    cert = []
    for t in range(T):
        lt = logits_all[:, t, :]
        ce.append(F.cross_entropy(lt, y, reduction="none"))
        cert.append(certainty_from_logits(lt))
    ce = torch.stack(ce, dim=1)      # [B,T]
    cert = torch.stack(cert, dim=1)  # [B,T]

    t_min = ce.argmin(dim=1)
    t_maxc = cert.argmax(dim=1)

    idx = torch.arange(B, device=logits_all.device)
    loss_min = ce[idx, t_min].mean()
    loss_maxc = ce[idx, t_maxc].mean()
    total = 0.5 * (loss_min + loss_maxc)
    return total, loss_min.detach(), loss_maxc.detach()


def loss_monotone_cls(logits_all: torch.Tensor, y: torch.Tensor, lambda_mono: float, margin: float):
    """
    total = CE(final) + lambda * mean ReLU(CE_{t+1}-CE_t + margin)
    returns (total, aux1=finalCE, aux2=mono_pen)
    """
    B, T, C = logits_all.shape
    ce = torch.stack(
        [F.cross_entropy(logits_all[:, t, :], y, reduction="none") for t in range(T)],
        dim=1,
    )  # [B,T]

    final_ce = ce[:, -1].mean()
    diffs = ce[:, 1:] - ce[:, :-1] + float(margin)
    mono = F.relu(diffs).mean()
    total = final_ce + float(lambda_mono) * mono
    return total, final_ce.detach(), mono.detach()


def loss_allticks_monotone_cls(logits_all: torch.Tensor, y: torch.Tensor, lambda_mono: float, margin: float):
    """
    total = mean_t CE_t + lambda * mean ReLU(CE_{t+1}-CE_t + margin)
    returns (total, aux1=meanCE_allticks, aux2=mono_pen)
    """
    B, T, C = logits_all.shape
    ce = torch.stack(
        [F.cross_entropy(logits_all[:, t, :], y, reduction="none") for t in range(T)],
        dim=1,
    )  # [B,T]

    mean_ce = ce.mean()
    diffs = ce[:, 1:] - ce[:, :-1] + float(margin)
    mono = F.relu(diffs).mean()
    total = mean_ce + float(lambda_mono) * mono
    return total, mean_ce.detach(), mono.detach()

