from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrainConfig:
    # System
    device: str = "cuda"                 # "cpu" | "cuda"
    amp: bool = True
    seed: int = 0

    # Data
    dataset: str = "cifar10"             # "cifar10" | "cifar100"
    data_dir: str = "./data_cifar"
    num_workers: int = 4
    val_size: int = 5000                 # held out from the training set; 0 => validate on test (legacy)
    eval_test: bool = True               # evaluate the best checkpoint on the test set once, at the end

    # Training
    epochs: int = 50
    batch_size: int = 256
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    steps_per_epoch: int = 0             # 0 => full loader

    # Model
    model_type: str = "ctm"              # "ctm" | "cnn" (same backbone + linear head baseline)
    D: int = 256
    M: int = 32
    H: int = 256
    T_train: int = 10
    T_eval_mult: int = 2                  # eval ticks = T_train*T_eval_mult

    feat_dim: int = 64
    attn_dim: int = 64
    attn_heads: int = 16

    Dout_pairs: int = 512
    Dact_pairs: int = 512
    self_pairs_out: int = 0
    self_pairs_act: int = 0
    pair_seed: int = 123

    syn_depth: int = 10
    syn_bottleneck: int = 16
    dropout: float = 0.0

    # CTM dynamics knobs
    sync_after_update: bool = True      # read output synchronisation after the tick's state update (as in
                                        # the CTM paper). False reproduces <=0.1 runs, where tick-0 logits
                                        # were identical for every image.
    z_residual: bool = True
    eta: float = 0.1
    layernorm_z: bool = True            # apply LN to z every tick
    layernorm_o: bool = True            # apply LN on (o+q) residual
    beta_per_sample: bool = False       # normalize pairs with per-sample beta

    # Feature mode
    state_feat: bool = False            # state-dependent feature extractor per tick
    hyper_rank: int = 16
    hyper_hidden: int = 128

    # Logits refinement
    additive_logits: bool = False       # logits(t) = sum_{k<=t} delta(k)

    # Objectives
    loss_type: str = "minmax"           # "minmax" | "monotone" | "allticks" | "stochastic"
    eval_rule: str = "certainty"        # "final" | "certainty"
    lambda_mono: float = 0.05
    mono_margin: float = 0.0

    # Stochastic stopping (if loss_type=="stochastic")
    stoch_truncate_compute: bool = False

    # Speed
    compile: bool = False               # torch.compile the model (CUDA recommended)
    channels_last: bool = False

    # Adaptive compute analysis (certainty thresholds for early exit, evaluated at the end)
    halt_thresholds: tuple = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)

    # Logging / outputs
    save_checkpoints: bool = True       # best.pt (by validation) and last.pt in the run folder
    out_root: str = "./runs"
    run_name: str = ""                  # optional human tag

    # Tracking
    track_every: int = 50               # per-tick tracking on minibatches
    log_every: int = 50                 # global CSV / jsonl frequency

    # Plotting options
    keep_topk_cv: int = 6               # for cv_figure

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def T_eval(self) -> int:
        return int(self.T_train * self.T_eval_mult)

    def stable_run_id(self) -> str:
        """
        Stable hash based on full config (excluding run_name).
        """
        d = self.to_dict()
        # remove fields that should not affect identity
        for key in ("run_name", "num_workers", "compile", "save_checkpoints", "track_every", "log_every",
                    "out_root", "data_dir", "device"):
            d.pop(key, None)
        s = json.dumps(d, sort_keys=True)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def as_pretty_json(d: dict[str, Any]) -> str:
    return json.dumps(d, indent=2, sort_keys=True)

