import importlib.util
import pathlib
import subprocess
import sys

import pytest
import torch

from ctm_cifar.config import TrainConfig
from ctm_cifar.model import CTMCIFARClassifier, CTMOptions
from ctm_cifar.train import build_model

SMALL = dict(D=48, M=6, H=16, T_train=3, Dout_pairs=48, Dact_pairs=48, syn_depth=2, attn_heads=4, attn_dim=32)


def _model(**kw):
    torch.manual_seed(0)
    return build_model(TrainConfig(**{**SMALL, **kw}), 10).eval()


@pytest.mark.parametrize("state_feat", [False, True])
def test_tick0_depends_on_image(state_feat):
    x = torch.randn(6, 3, 32, 32)
    with torch.no_grad():
        new = _model(state_feat=state_feat, hyper_rank=4, hyper_hidden=16)(x)
        old = _model(state_feat=state_feat, hyper_rank=4, hyper_hidden=16, sync_after_update=False)(x)
    assert (new[:, 0] - new[:1, 0]).abs().max() > 1e-4
    # legacy order: the first tick's logits cannot see the image
    assert (old[:, 0] - old[:1, 0]).abs().max() < 1e-6


def test_shapes_and_max_ticks():
    m = _model()
    x = torch.randn(2, 3, 32, 32)
    assert m(x).shape == (2, 6, 10)  # T_eval = 2 * T_train
    assert m(x, max_ticks=2).shape == (2, 2, 10)
    assert m(x, max_ticks=99).shape == (2, 6, 10)
    assert _model(model_type="cnn")(x).shape == (2, 1, 10)


def test_additive_logits_are_cumulative():
    m = _model()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        delta = m(x)
        cum = m(x, additive_logits=True)
    torch.testing.assert_close(cum, delta.cumsum(dim=1))


def _legacy_module():
    """model.py as of the first commit, if git history is available."""
    try:
        src = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%H", "--", "src/ctm_cifar/model.py"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        code = subprocess.run(["git", "show", f"{src[-1]}:src/ctm_cifar/model.py"],
                              capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return None
    path = pathlib.Path(__file__).parent / "_legacy_model_tmp.py"
    path.write_text(code)
    try:
        spec = importlib.util.spec_from_file_location("_legacy_model_tmp", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        path.unlink()
    return mod


def test_legacy_flag_matches_original_code():
    legacy = _legacy_module()
    if legacy is None:
        pytest.skip("no git history (shallow checkout)")
    kw = dict(D=48, M=6, H=16, T_ticks=5, feat_dim=64, attn_dim=32, attn_heads=4, Dout_pairs=48,
              Dact_pairs=48, self_pairs_out=4, self_pairs_act=4, pair_seed=1, syn_depth=2,
              syn_bottleneck=8, dropout=0.0, z_residual=True, eta=0.1, layernorm_z=True,
              layernorm_o=True, beta_per_sample=True, state_feat=False, hyper_rank=4, hyper_hidden=16)
    torch.manual_seed(0)
    a = legacy.CTMCIFARClassifier(10, legacy.CTMOptions(**kw)).eval()
    b = CTMCIFARClassifier(10, CTMOptions(**kw, sync_after_update=False)).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(a(x), b(x))
