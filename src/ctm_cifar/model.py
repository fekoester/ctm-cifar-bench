from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_pairs(num_neurons: int, num_pairs: int, num_self: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(int(seed))

    num_self = min(int(num_self), int(num_neurons), int(num_pairs))
    self_idx = torch.randperm(num_neurons, generator=g)[:num_self]
    i_self = self_idx
    j_self = self_idx

    rem = int(num_pairs) - num_self
    if rem > 0:
        i_rand = torch.randint(0, num_neurons, (rem,), generator=g)
        j_rand = torch.randint(0, num_neurons, (rem,), generator=g)
        i = torch.cat([i_self, i_rand], dim=0)
        j = torch.cat([j_self, j_rand], dim=0)
    else:
        i, j = i_self, j_self
    return i, j


class BasicBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        C = int(channels)
        self.conv1 = nn.Conv2d(C, C, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(C)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class CIFARConvBackbone(nn.Module):
    """
    Minimal CIFAR backbone. Output: [B, feat_dim(=64), 16, 16]
    """
    def __init__(self, feat_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        C = int(feat_dim)
        if C != 64:
            raise ValueError("This backbone expects feat_dim=64 (easy to extend if you want).")

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, 2, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        self.block1 = BasicBlock(C)
        self.block2 = BasicBlock(C)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.drop(h)


class TokensFromMap(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.ln = nn.LayerNorm(int(C))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        tokens = h.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return self.ln(tokens)


class UNetMLP(nn.Module):
    """
    U-shaped MLP with skip connections.
    depth must be even.
    """
    def __init__(self, in_dim: int, out_dim: int, depth: int = 10, bottleneck: int = 16):
        super().__init__()
        depth = int(depth)
        assert depth % 2 == 0, "UNetMLP depth must be even"
        out_dim = int(out_dim)
        in_dim = int(in_dim)
        bottleneck = int(bottleneck)

        down_layers = depth // 2
        up_layers = depth // 2

        self.in_proj = nn.Linear(in_dim, out_dim)

        down_widths = [
            round(out_dim + (bottleneck - out_dim) * (k + 1) / down_layers)
            for k in range(down_layers)
        ]
        up_widths = [
            round(bottleneck + (out_dim - bottleneck) * (k + 1) / up_layers)
            for k in range(up_layers)
        ]

        self.enc = nn.ModuleList()
        prev = out_dim
        for w in down_widths:
            self.enc.append(nn.Linear(prev, w))
            prev = w

        self.dec = nn.ModuleList()
        prev = down_widths[-1]
        for w in up_widths:
            self.dec.append(nn.Linear(prev, w))
            prev = w

        self.skip_norms = nn.ModuleList([nn.LayerNorm(down_widths[k]) for k in range(down_layers)])
        self.skip_adapters = nn.ModuleList()
        for k in range(down_layers):
            enc_w = down_widths[k]
            dec_w = up_widths[up_layers - 1 - k]
            self.skip_adapters.append(nn.Identity() if enc_w == dec_w else nn.Linear(enc_w, dec_w))

        self.out_proj = nn.Linear(up_widths[-1], out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.in_proj(x))
        skips = []
        for layer in self.enc:
            h = F.gelu(layer(h))
            skips.append(h)

        for idx, layer in enumerate(self.dec):
            h = F.gelu(layer(h))
            enc_idx = len(self.enc) - 1 - idx
            s = self.skip_norms[enc_idx](skips[enc_idx])
            s = self.skip_adapters[enc_idx](s)
            if s.shape[-1] == h.shape[-1]:
                h = h + s

        return self.out_proj(h)


class PerNeuronDepth1MLP(nn.Module):
    """
    Per-neuron MLP reading action history A: [B,D,M] -> dz: [B,D]
    """
    def __init__(self, D: int, M: int, H: int, dropout: float = 0.0):
        super().__init__()
        D, M, H = int(D), int(M), int(H)
        self.w1 = nn.Parameter(torch.randn(D, M, H) * (1.0 / math.sqrt(max(1, M))))
        self.b1 = nn.Parameter(torch.zeros(1, D, H))
        self.w2 = nn.Parameter(torch.randn(D, H) * (1.0 / math.sqrt(max(1, H))))
        self.b2 = nn.Parameter(torch.zeros(1, D))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("bdm,dmh->bdh", A, self.w1) + self.b1
        h = F.gelu(h)
        h = self.drop(h)
        z = torch.einsum("bdh,dh->bd", h, self.w2) + self.b2
        return z


class HyperChannelMix(nn.Module):
    """
    Low-rank hypernetwork dynamic 1x1 channel mixing on feature maps.
    """
    def __init__(self, C: int, D: int, rank: int = 16, hidden: int = 128):
        super().__init__()
        C, D, rank, hidden = int(C), int(D), int(rank), int(hidden)
        self.C = C
        self.D = D
        self.rank = rank

        self.U = nn.Parameter(torch.randn(C, rank))
        self.V = nn.Parameter(torch.randn(rank, C))

        self.scale_net = nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, rank))
        self.bias_net = nn.Sequential(nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, C))

        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        assert C == self.C
        hw = H * W
        h_flat = h.view(B, C, hw)  # [B,C,HW]
        tmp = torch.einsum("rc,bch->brh", self.V, h_flat)  # [B,R,HW]
        s = self.scale_net(z)  # [B,R]
        tmp = tmp * s.unsqueeze(-1)
        out = torch.einsum("cr,brh->bch", self.U, tmp).view(B, C, H, W)
        b = self.bias_net(z).view(B, C, 1, 1)
        return h + self.gate * out + b


@dataclass
class CTMOptions:
    D: int
    M: int
    H: int
    T_ticks: int
    feat_dim: int
    attn_dim: int
    attn_heads: int
    Dout_pairs: int
    Dact_pairs: int
    self_pairs_out: int
    self_pairs_act: int
    pair_seed: int
    syn_depth: int
    syn_bottleneck: int
    dropout: float

    # knobs
    z_residual: bool
    eta: float
    layernorm_z: bool
    layernorm_o: bool
    beta_per_sample: bool
    state_feat: bool
    hyper_rank: int
    hyper_hidden: int
    # Output synchronisation read after the tick's z update (CTM paper). False: legacy order, where the
    # logits of tick t only see z from before tick t, so tick-0 logits are the same for every image.
    sync_after_update: bool = True


class CTMCIFARClassifier(nn.Module):
    def __init__(self, num_classes: int, opt: CTMOptions):
        super().__init__()
        self.num_classes = int(num_classes)
        self.opt = opt
        self.D = int(opt.D)
        self.M = int(opt.M)
        self.T_ticks = int(opt.T_ticks)
        self.attn_dim = int(opt.attn_dim)

        self.backbone = CIFARConvBackbone(feat_dim=opt.feat_dim, dropout=opt.dropout)
        self.tokenizer = TokensFromMap(C=opt.feat_dim)

        self.state_feat = bool(opt.state_feat)
        if self.state_feat:
            self.hyperconv = HyperChannelMix(
                C=opt.feat_dim, D=opt.D, rank=opt.hyper_rank, hidden=opt.hyper_hidden
            )

        self.k_proj = nn.Linear(opt.feat_dim, opt.attn_dim)
        self.v_proj = nn.Linear(opt.feat_dim, opt.attn_dim)
        self.q_proj = nn.Linear(opt.attn_dim, opt.attn_dim)

        self.attn = nn.MultiheadAttention(opt.attn_dim, opt.attn_heads, batch_first=True)

        self.synapse = UNetMLP(
            in_dim=opt.D + opt.attn_dim,
            out_dim=opt.D,
            depth=opt.syn_depth,
            bottleneck=opt.syn_bottleneck,
        )
        self.nlm = PerNeuronDepth1MLP(D=opt.D, M=opt.M, H=opt.H, dropout=opt.dropout)

        self.z0 = nn.Parameter(torch.zeros(opt.D))
        self.A0 = nn.Parameter(torch.zeros(opt.D, opt.M))

        i_out, j_out = make_pairs(opt.D, opt.Dout_pairs, opt.self_pairs_out, seed=opt.pair_seed)
        i_act, j_act = make_pairs(opt.D, opt.Dact_pairs, opt.self_pairs_act, seed=opt.pair_seed + 1)
        self.register_buffer("i_out", i_out)
        self.register_buffer("j_out", j_out)
        self.register_buffer("i_act", i_act)
        self.register_buffer("j_act", j_act)

        self.r_out_param = nn.Parameter(torch.zeros(opt.Dout_pairs))
        self.r_act_param = nn.Parameter(torch.zeros(opt.Dact_pairs))

        self.W_out = nn.Linear(opt.Dout_pairs, self.num_classes)
        self.W_in = nn.Linear(opt.Dact_pairs, opt.attn_dim)

        # residual merge o+q optional LN
        self.o_ln = nn.LayerNorm(opt.attn_dim)

        self.eta = nn.Parameter(torch.tensor(float(opt.eta)))

    def _sync_out(self, z, alpha_out, beta_out, decay_out):
        """Decayed running synchronisation of the output neuron pairs."""
        zi_out = z[:, self.i_out]
        zj_out = z[:, self.j_out]
        alpha_out = alpha_out * decay_out.unsqueeze(0) + (zi_out * zj_out)
        if self.opt.beta_per_sample:
            beta_out = beta_out * decay_out.unsqueeze(0) + 1.0
            S_out = alpha_out / torch.sqrt(beta_out.clamp_min(1e-8))
        else:
            beta_out = beta_out * decay_out + 1.0
            S_out = alpha_out / torch.sqrt(beta_out.clamp_min(1e-8)).unsqueeze(0)
        return alpha_out, beta_out, S_out

    def _maybe_ln_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.opt.layernorm_z:
            return F.layer_norm(z, (z.shape[-1],))
        return z

    def _maybe_ln_o(self, o: torch.Tensor) -> torch.Tensor:
        if self.opt.layernorm_o:
            return self.o_ln(o)
        return o

    def forward(self, x_img: torch.Tensor, max_ticks: int | None = None, additive_logits: bool = False):
        B = x_img.size(0)
        device = x_img.device

        # feature map once
        h_base = self.backbone(x_img)

        # baseline fixed K,V (if not state_feat)
        if not self.state_feat:
            tokens0 = self.tokenizer(h_base)    # [B,L,C]
            K0 = self.k_proj(tokens0)
            V0 = self.v_proj(tokens0)

        z = self.z0.unsqueeze(0).expand(B, -1)
        A = self.A0.unsqueeze(0).expand(B, -1, -1).contiguous()

        # running stats for pairs
        Pout = self.i_out.numel()
        Pact = self.i_act.numel()

        alpha_out = torch.zeros(B, Pout, device=device)
        alpha_act = torch.zeros(B, Pact, device=device)

        if self.opt.beta_per_sample:
            beta_out = torch.zeros(B, Pout, device=device)
            beta_act = torch.zeros(B, Pact, device=device)
        else:
            beta_out = torch.zeros(Pout, device=device)
            beta_act = torch.zeros(Pact, device=device)

        r_out = F.softplus(self.r_out_param)
        r_act = F.softplus(self.r_act_param)
        decay_out = torch.exp(-r_out)  # [Pout]
        decay_act = torch.exp(-r_act)

        T_run = self.T_ticks if max_ticks is None else max(1, min(int(max_ticks), self.T_ticks))

        logits_all = []
        logits_cum = torch.zeros(B, self.num_classes, device=device)

        for _t in range(T_run):
            if self.state_feat:
                h_t = self.hyperconv(h_base, z)
                tokens = self.tokenizer(h_t)
                K = self.k_proj(tokens)
                V = self.v_proj(tokens)
            else:
                K, V = K0, V0

            # Action synchronisation (drives the attention query) from the current state.
            zi_act = z[:, self.i_act]
            zj_act = z[:, self.j_act]
            if self.opt.beta_per_sample:
                alpha_act = alpha_act * decay_act.unsqueeze(0) + (zi_act * zj_act)
                beta_act = beta_act * decay_act.unsqueeze(0) + 1.0
                S_act = alpha_act / torch.sqrt(beta_act.clamp_min(1e-8))
            else:
                alpha_act = alpha_act * decay_act.unsqueeze(0) + (zi_act * zj_act)
                beta_act = beta_act * decay_act + 1.0
                S_act = alpha_act / torch.sqrt(beta_act.clamp_min(1e-8)).unsqueeze(0)

            if not self.opt.sync_after_update:
                alpha_out, beta_out, S_out = self._sync_out(z, alpha_out, beta_out, decay_out)
                delta_logits = self.W_out(S_out)

            q = self.W_in(S_act)                    # [B,attn_dim]
            Q = self.q_proj(q).unsqueeze(1)         # [B,1,attn_dim]

            o, _ = self.attn(Q, K, V, need_weights=False)
            o_t = o.squeeze(1)                      # [B,attn_dim]
            o_t = self._maybe_ln_o(o_t + q)

            a_t = self.synapse(torch.cat([z, o_t], dim=-1))
            A = torch.roll(A, shifts=-1, dims=2)
            A[:, :, -1] = a_t

            dz = self.nlm(A)

            if self.opt.z_residual:
                z = z + self.eta * dz
            else:
                z = dz

            z = self._maybe_ln_z(z)

            if self.opt.sync_after_update:
                # Output synchronisation from the updated state: tick t's logits use tick t's read.
                alpha_out, beta_out, S_out = self._sync_out(z, alpha_out, beta_out, decay_out)
                delta_logits = self.W_out(S_out)

            if additive_logits:
                logits_cum = logits_cum + delta_logits
                logits_all.append(logits_cum)
            else:
                logits_all.append(delta_logits)

        return torch.stack(logits_all, dim=1)  # [B,T,C]



class CNNBaseline(nn.Module):
    """
    Reference model: the same convolutional backbone, global average pooling and a linear head.
    Returns logits shaped [B, 1, C] so it plugs into the multi-tick training/evaluation code.
    It answers what the CTM's recurrent ticks add on top of the backbone.
    """

    def __init__(self, num_classes: int, feat_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.backbone = CIFARConvBackbone(feat_dim=feat_dim, dropout=dropout)
        self.head = nn.Linear(feat_dim, int(num_classes))
        self.T_ticks = 1

    def forward(self, x_img: torch.Tensor, max_ticks: int | None = None, additive_logits: bool = False):
        h = self.backbone(x_img).mean(dim=(2, 3))
        return self.head(h).unsqueeze(1)
