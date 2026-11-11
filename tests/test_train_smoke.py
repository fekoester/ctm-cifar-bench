import json
import os
import sys

import pytest

from ctm_cifar.config import TrainConfig
from ctm_cifar.train import load_checkpoint, train_one


@pytest.mark.parametrize("model_type,loss", [("ctm", "minmax"), ("ctm", "stochastic"), ("cnn", "minmax")])
def test_end_to_end_fake_data(tmp_path, model_type, loss):
    cfg = TrainConfig(
        device="cpu", dataset="fake", val_size=64, num_workers=0, epochs=2, batch_size=32,
        D=32, M=4, H=16, T_train=2, Dout_pairs=32, Dact_pairs=32, syn_depth=2, attn_heads=4,
        attn_dim=32, model_type=model_type, loss_type=loss, out_root=str(tmp_path), track_every=2,
        log_every=2,
    )
    run_dir = train_one(cfg)
    final = json.load(open(os.path.join(run_dir, "final_metrics.json")))
    assert final["best_epoch"] in (1, 2)
    assert set(final["test"]["rules"]) == {"final", "certainty", "best_tick", "mean_logits"}
    model, cfg2, ckpt = load_checkpoint(os.path.join(run_dir, "best.pt"))
    assert cfg2.model_type == model_type and ckpt["epoch"] == final["best_epoch"]


def test_cli_sweep_expansion():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from train_ctm_cifar import build_parser, expand_runs

    args = build_parser().parse_args(["--sweep4", "--lambda_sweep", "0,0.1", "--amp"])
    runs = expand_runs(args)
    assert len(runs) == 6  # per z_residual: minmax + 2 monotone lambdas
    assert all(r.amp for r in runs)
    args = build_parser().parse_args(["--amp", "0", "--sync_after_update", "0"])
    (r,) = expand_runs(args)
    assert not r.amp and not r.sync_after_update
