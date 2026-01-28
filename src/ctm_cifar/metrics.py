from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def normalized_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    p: [..., C] probabilities
    returns entropy/log(C)
    """
    C = p.size(-1)
    p = p.clamp_min(eps)
    ent = -(p * p.log()).sum(dim=-1)
    return ent / math.log(C)


def certainty_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    return 1.0 - normalized_entropy(p)


@torch.no_grad()
def per_tick_metrics_from_logits(logits_all: torch.Tensor, y: torch.Tensor):
    """
    logits_all: [B,T,C]
    y: [B]
    returns loss_t, acc_t, cert_t as [T]
    """
    B, T, C = logits_all.shape
    loss_list, acc_list, cert_list = [], [], []
    for t in range(T):
        lt = logits_all[:, t, :]
        ce = F.cross_entropy(lt, y, reduction="none")
        pred = lt.argmax(dim=-1)
        acc = (pred == y).float()
        cert = certainty_from_logits(lt)
        loss_list.append(ce.mean())
        acc_list.append(acc.mean())
        cert_list.append(cert.mean())
    return torch.stack(loss_list), torch.stack(acc_list), torch.stack(cert_list)

