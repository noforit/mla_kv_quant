"""Shared orthogonal updates in the H_K + H_V eigenspaces."""

import torch
from torch import nn
from .quantizer import skew_exp_rotation


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


def projection_basis(base, attention):
    hk, hv = projection_grams(attention)
    r = base.detach().cpu().double()
    metric = r.T @ (hk.cpu().double() + hv.cpu().double()) @ r
    _, basis = torch.linalg.eigh((metric + metric.T) / 2)
    basis = basis.flip(-1)
    pivots = basis.abs().argmax(dim=0)
    signs = basis[pivots, torch.arange(basis.shape[1])].sign()
    return (basis * signs).to(base.device, torch.float32)


class SharedRotation(nn.Module):
    def __init__(self, base, attention):
        super().__init__()
        self.register_buffer("base", base.detach().float().clone())
        width = base.shape[0]
        dimensions = (width // 2, width - width // 2)
        self.raw = nn.ParameterList([nn.Parameter(base.new_zeros(d, d)) for d in dimensions])
        self.register_buffer("basis", projection_basis(base, attention))
        self.register_buffer("identity", torch.eye(width, device=base.device))

    def forward(self):
        factors = [skew_exp_rotation(torch.eye(p.shape[0], device=p.device), p) for p in self.raw]
        update = torch.block_diag(*factors) - self.identity
        return self.base @ (self.identity + self.basis @ update @ self.basis.T)
