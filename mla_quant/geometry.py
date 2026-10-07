"""The four shared-latent rotation settings used in the paper experiments."""

import torch
from torch import nn
from .quantizer import skew_exp_rotation, skew_cayley_rotation

SETTINGS = ("full_joint", "kv_energy", "kv_weight", "cayley")


def projection_grams(attention):
    config = attention.config
    weight = attention.kv_b_proj.weight.detach().float()
    blocks = weight.reshape(
        config.num_attention_heads, config.qk_nope_head_dim + config.v_head_dim, config.kv_lora_rank
    )
    wk = blocks[:, : config.qk_nope_head_dim].reshape(-1, config.kv_lora_rank)
    wv = blocks[:, config.qk_nope_head_dim :].reshape(-1, config.kv_lora_rank)
    grams = [w.T @ w for w in (wk, wv)]
    return tuple(h / h.trace() for h in grams)


def projection_basis(base, attention, setting):
    hk, hv = projection_grams(attention)
    metric = hk + hv if setting == "kv_energy" else hk - hv
    r = base.detach().cpu().double()
    metric = r.T @ metric.detach().cpu().double() @ r
    _, basis = torch.linalg.eigh((metric + metric.T) / 2)
    basis = basis.flip(-1)  # Descending SIGNED eigenvalue for H_K-H_V.
    pivots = basis.abs().argmax(dim=0)
    signs = basis[pivots, torch.arange(basis.shape[1])].sign()
    return (basis * signs).to(base.device, torch.float32)


class SharedRotation(nn.Module):
    def __init__(self, base, setting, attention=None):
        super().__init__()
        if setting not in SETTINGS:
            raise ValueError(setting)
        self.setting = setting
        self.register_buffer("base", base.detach().float().clone())
        width = base.shape[0]
        structured = setting in ("kv_energy", "kv_weight")
        dimensions = (width // 2, width // 2) if structured else (width,)
        self.raw = nn.ParameterList([nn.Parameter(base.new_zeros(d, d)) for d in dimensions])
        if structured:
            self.register_buffer("basis", projection_basis(base, attention, setting))
            self.register_buffer("identity", torch.eye(width, device=base.device))

    def forward(self):
        if self.setting == "cayley":
            return skew_cayley_rotation(self.base, self.raw[0])
        if self.setting == "full_joint":
            return skew_exp_rotation(self.base, self.raw[0])
        factors = [skew_exp_rotation(torch.eye(p.shape[0], device=p.device), p) for p in self.raw]
        update = torch.block_diag(*factors) - self.identity
        return self.base @ (self.identity + self.basis @ update @ self.basis.T)
