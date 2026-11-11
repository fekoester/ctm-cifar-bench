import torch

from ctm_cifar.data import split_indices
from ctm_cifar.eval import adaptive_halting_curve, choose_best_tick, select_logits


def _logits():
    # 3 samples, 3 ticks, 2 classes. True label 0 for all.
    lg = torch.tensor([
        [[0.0, 1.0], [3.0, 0.0], [0.0, 0.1]],   # confident & right at t=1
        [[0.0, 0.2], [0.1, 0.0], [5.0, 0.0]],   # right & confident at t=2
        [[4.0, 0.0], [0.0, 0.1], [0.0, 0.2]],   # confident & right at t=0
    ])
    return lg, torch.zeros(3, dtype=torch.long)


def test_rules():
    lg, y = _logits()
    assert select_logits(lg, "final").argmax(-1).tolist() == [1, 0, 1]
    assert select_logits(lg, "certainty").argmax(-1).tolist() == [0, 0, 0]
    assert select_logits(lg, "best_tick", 1).argmax(-1).tolist() == [0, 0, 1]
    assert choose_best_tick(lg, y) in (0, 1, 2)
    assert select_logits(lg, "mean_logits").shape == (3, 2)


def test_halting_curve():
    lg, y = _logits()
    curve = adaptive_halting_curve(lg, y, [0.0, 0.5, 1.01])
    assert curve[0]["mean_ticks"] == 1.0  # threshold 0 halts at the first tick
    assert curve[-1]["mean_ticks"] == 3.0  # unreachable threshold runs all ticks
    assert curve[1]["acc"] == 1.0


def test_split_is_disjoint_and_deterministic():
    a, b = split_indices(100, 20, seed=3)
    assert len(a) == 80 and len(b) == 20 and not set(a) & set(b)
    assert (a, b) == split_indices(100, 20, seed=3)
