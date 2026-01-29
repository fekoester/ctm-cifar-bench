# CTM-CIFAR (Multi-tick refinement for CIFAR-10/100)

Implementation of a CTM-style multi-tick refinement classifier for CIFAR-10 / CIFAR-100 with:

- Multi-tick logits (T_train ticks for training, T_eval_mult * T_train ticks for eval)
- Multiple objective options:
  - minmax: 0.5 * (min CE over ticks + CE at max-certainty tick)
  - monotone: CE(final) + λ · mean ReLU(CE(t+1) − CE(t) + margin)
  - allticks: mean CE over all ticks + λ · monotone penalty
  - stochastic: sample a stop tick N and train only that tick
- Deterministic, stable run folders with per-run JSON logs + global runs.csv
- Per-tick tracking and summary plotting scripts

This repo is meant to be an implementation of Sakana AI's continuous thought machines with the addition of a residual connection and optional normalization in the neuron synchronization step, while also adding different loss functions.

## Project structure

    ctm-cifar/
      src/ctm_cifar/
        __init__.py
        config.py
        data.py
        model.py
        losses.py
        metrics.py
        logging_utils.py
        train.py
        eval.py
        plotting.py
      scripts/
        train_ctm_cifar.py
        plot_summary.py
      README.md
      pyproject.toml
      LICENSE
      .gitignore


## Installation

Option A: editable install (recommended)

    python -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install -e .

Option B: minimal deps (no packaging)

    pip install torch torchvision matplotlib tqdm numpy


## Quickstart

Single run (CIFAR-10)

    python scripts/train_ctm_cifar.py       --device cuda       --dataset cifar10       --epochs 50       --batch_size 256       --loss_type minmax       --eval_rule certainty       --z_residual 1       --amp

Outputs a run folder under:

    runs_cifar_ctm/<timestamp_or_run_name>_.../

with:

- config.json
- model_stats.json
- history.json
- eval_per_tick.json
- tick_tracking.pt
- plots (per-run)

Plus a global runs.csv in runs_cifar_ctm/.


## Canonical sweeps

4-run sweep: minmax/monotone × zres off/on

    python scripts/train_ctm_cifar.py       --device cuda       --dataset cifar10       --epochs 50       --batch_size 256       --sweep4       --lambda_mono 0.2       --amp

Lambda sweep for monotone (within sweep4)

    python scripts/train_ctm_cifar.py       --device cuda       --dataset cifar10       --epochs 30       --batch_size 256       --sweep4       --lambda_sweep 0,0.01,0.02,0.05,0.1,0.2,0.3       --amp


## Plotting sweep summaries

After runs exist:

    python scripts/plot_summary.py --out_root ./runs_cifar_ctm

This writes to:

    runs_cifar_ctm/summary_plots/

Main figures:

- val_acc_vs_epoch.png
- val_ce_vs_epoch.png
- val_acc_besttick_vs_epoch.png
- val_acc_certtick_vs_epoch.png
- eval_per_tick_acc.png
- eval_per_tick_ce.png
- eval_per_tick_certainty.png
- cv_figure.png

Filter by dataset:

    python scripts/plot_summary.py --out_root ./runs_cifar_ctm --dataset cifar10


## Key flags (training)

Common:

- --dataset {cifar10,cifar100}
- --device {cpu,cuda}
- --amp (bfloat16 autocast + GradScaler on CUDA)
- --epochs, --batch_size, --grad_accum
- --lr, --weight_decay, --warmup_steps
- --loss_type {minmax,monotone,allticks,stochastic}
- --eval_rule {final,certainty}
- --T_train, --T_eval_mult

Model size:

- --D : action/state dim
- --M : FIFO history length
- --H : per-neuron MLP hidden size
- --attn_dim, --attn_heads
- --Dout_pairs, --Dact_pairs
- --syn_depth, --syn_bottleneck
- --dropout

Ablations / compute:

- --z_residual {0,1} : residual update z ← z + η·dz vs overwrite
- --additive_logits : logits(t) = sum_{k<=t} delta_logits(k) (cumulative refinement)
- --state_feat {0,1} : fixed tokens vs state-dependent features per tick
- --hyper_rank, --hyper_hidden : hypernetwork settings (when --state_feat 1)
- --stoch_truncate_compute : in stochastic mode, only run N ticks to save compute
- --track_every : per-tick tracking cadence

Monotone loss:

- --lambda_mono
- --mono_margin
- --lambda_sweep "0,0.05,0.1,0.2"


## Reproducibility

- The run folder always contains a full config.json.
- The run ID is a stable hash of the config (excluding run_tag).
- Seeds set both torch and numpy (and DataLoader generator).


## Results

End-of-run **per-tick validation accuracy** on CIFAR-10 for three training objectives (**minmax**, **monotone**, **stochastic**), each with and without the *z-residual* connection (`zres=0` solid, `zres=1` dashed).
Training was done just for T=25 ticks, but evaluation was conducted up to 50 ticks.

Key takeaways from this run:
- **Stochastic** training reaches the best and most stable performance across ticks (≈ low-80% and flat).
- **Monotone** improves up to a mid-tick peak but degrades at later ticks (collapse for `zres=0`, slower decay for `zres=1`).
- **Minmax** peaks around mid ticks but is noticeably less stable, and the residual variant underperforms here.

![End-of-run per-tick accuracy (CIFAR-10)](assets/eval_per_tick_acc.png)

Template:

- Dataset: CIFAR-10
- Model: D=256, M=32, H=256, T_train=10, T_eval=20
- Objective: minmax, z_residual=1, eval_rule=certainty
- Final acc @ 2T last tick: XX.XX%
- Best-tick acc (max over ticks): XX.XX%
- Best-tick acc (certainty tick): XX.XX%

Add the plot:

- summary_plots/cv_figure.png


## Citation / attribution

If you publish results using this code, cite the original CTM work.
Original paper: [arXiv:2505.05522](https://arxiv.org/abs/2505.05522)
