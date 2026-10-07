"""MLA content-cache QDQ, padding-aware sink/recent windows, and baselines."""

from __future__ import annotations
import functools
import importlib
import inspect
import os
import warnings
from collections.abc import Callable, Iterable
from typing import Any
import torch
from torch import nn
from transformers.cache_utils import (
    Cache,
    CacheLayerMixin,
    DynamicCache,
    get_layer_types_and_kwargs,
)
from .oscar import OscarMLAQuantizer

SUPPORTED_TRANSFORMERS_MODELS = {
    "deepseek_v2": ("transformers.models.deepseek_v2.modeling_deepseek_v2", "DeepseekV2Model"),
    "deepseek_v3": ("transformers.models.deepseek_v3.modeling_deepseek_v3", "DeepseekV3Model"),
    "minicpm3": ("transformers.models.minicpm3.modeling_minicpm3", "MiniCPM3Model"),
}
_CLASS_PATCH_STATE = "_mla_kv_fake_quant_patch_state"
_ACTIVE_CLASS_QUANTIZERS: dict[str, "FakeIntQuantizer | OscarMLAQuantizer"] = {}
_PENDING_PAD_MASKS: list[torch.Tensor] = []
_PAD_MASK_STATS = {
    "masks_published": 0,
    "forwards_with_candidate_mask": 0,
    "forwards_where_padding_was_excluded": 0,
}
_WINDOW_STATS = {
    "updates": 0,
    "high_precision_rows": 0,
    "quantized_rows": 0,
    "transitions_hp_to_quantized": 0,
    "requantized_rows": 0,
}
_WINDOW_PENDING: dict[str, torch.Tensor | None] = {
    "high_precision_rows": None,
    "quantized_rows": None,
    "transitions_hp_to_quantized": None,
    "requantized_rows": None,
}


def _pending_window_add(name: str, value: torch.Tensor) -> None:
    current = _WINDOW_PENDING[name]
    _WINDOW_PENDING[name] = value if current is None else current + value


def _flush_window_stats() -> None:
    for name, value in list(_WINDOW_PENDING.items()):
        if value is not None:
            _WINDOW_STATS[name] += int(value.item())
            _WINDOW_PENDING[name] = None


_LAYER_CALL_STATS: dict[str, Any] = {"per_layer_calls": {}, "prefill_calls": 0, "decode_calls": 0}


def _reset_runtime_stats() -> None:
    """Reset process-local counters before installing a fresh patch."""
    _PENDING_PAD_MASKS.clear()
    for key in _PAD_MASK_STATS:
        _PAD_MASK_STATS[key] = 0
    for key in _WINDOW_STATS:
        _WINDOW_STATS[key] = 0
    for key in _WINDOW_PENDING:
        _WINDOW_PENDING[key] = None
    _LAYER_CALL_STATS["per_layer_calls"].clear()
    _LAYER_CALL_STATS["prefill_calls"] = 0
    _LAYER_CALL_STATS["decode_calls"] = 0


def record_layer_call(layer_idx: int, tensor: torch.Tensor) -> None:
    """Count one INT2 quantization of one MLA layer's latent chunk."""
    try:
        index = int(layer_idx)
    except (TypeError, ValueError):
        return
    calls = _LAYER_CALL_STATS["per_layer_calls"]
    calls[index] = int(calls.get(index, 0)) + 1
    tokens = int(tensor.shape[-2]) if tensor.dim() >= 2 else 0
    if tokens <= 1:
        _LAYER_CALL_STATS["decode_calls"] = int(_LAYER_CALL_STATS["decode_calls"]) + 1
    else:
        _LAYER_CALL_STATS["prefill_calls"] = int(_LAYER_CALL_STATS["prefill_calls"]) + 1


def set_pending_pad_mask(mask: torch.Tensor | None) -> None:
    """Record a candidate valid-row mask for the next model forward.

    lm-eval pads inputs *and* continuations before one model call, so several
    masks can be recorded per forward; ``take_pending_pad_mask`` picks the one
    that matches the tokens the model is actually called with (the widest).
    """
    if mask is None:
        return
    _PENDING_PAD_MASKS.append(mask.detach().to("cpu", torch.bool))
    _PAD_MASK_STATS["masks_published"] += 1


def take_pending_pad_mask(tokens: int | None = None) -> torch.Tensor | None:
    """Consume the recorded masks so they can never leak into a later forward.

    ``tokens`` is the length the model is about to be called with; a mask of
    exactly that width is preferred, otherwise the widest one is used.
    """
    masks, _PENDING_PAD_MASKS[:] = (list(_PENDING_PAD_MASKS), [])
    if not masks:
        return None
    if tokens is not None:
        for mask in reversed(masks):
            if mask.shape[1] == tokens:
                return mask
    return max(masks, key=lambda item: item.shape[1])


def pad_fix_enabled() -> bool:
    """Whether padding is excluded from tile statistics.

    ``KVARN_LEGACY_PAD_STATS=1`` restores the historical behaviour where pad
    rows were part of a chunk's statistics. It gates *both* mask sources — the
    lm-eval adapter (loglikelihood) and the attention mask (generation) — so a
    legacy rerun really is the old numerical path and not a half-fixed one.
    """
    return str(os.environ.get("KVARN_LEGACY_PAD_STATS", "")).lower() not in {"1", "true", "yes"}


class FakeIntQuantizer:
    """Per-token, group-wise affine min-max fake quantizer.

    Quantization is performed along the final tensor dimension. INT2 uses all
    four codes in ``[0, 3]``. Each group has a floating-point scale and minimum;
    a final partial group is calibrated using only its real values.
    """

    def __init__(self, bits: int = 2, group_size: int = 128, int2_mode: str = "affine") -> None:
        if not isinstance(bits, int) or not 2 <= bits <= 8:
            raise ValueError(f"bits must be an integer in [2, 8], got {bits!r}")
        if not isinstance(group_size, int) or group_size <= 0:
            raise ValueError(f"group_size must be a positive integer, got {group_size!r}")
        from .oscar import INT2_MODES

        if int2_mode not in INT2_MODES:
            raise ValueError(f"int2_mode must be one of {INT2_MODES}, got {int2_mode!r}")
        if int2_mode != "affine" and bits != 2:
            raise ValueError("Lloyd-Max INT2 modes require bits=2")
        self.bits = bits
        self.group_size = group_size
        self.int2_mode = int2_mode
        self.qmin = 0
        self.qmax = (1 << bits) - 1
        self.call_count = 0
        self.element_count = 0
        self.first_shapes: list[tuple[int, ...]] = []

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_floating_point():
            raise TypeError(f"fake quantization expects a floating tensor, got {tensor.dtype}")
        if tensor.ndim == 0:
            raise ValueError("fake quantization expects a tensor with at least one dimension")
        width = tensor.shape[-1]
        if width == 0:
            raise ValueError("fake quantization does not support an empty final dimension")
        if tensor.numel() == 0:
            return tensor.clone()
        self.call_count += 1
        self.element_count += tensor.numel()
        if len(self.first_shapes) < 4:
            self.first_shapes.append(tuple(tensor.shape))
        values = tensor.float()
        full_width = width - width % self.group_size
        quantized_parts = []
        if full_width:
            full_groups = values[..., :full_width].reshape(*values.shape[:-1], -1, self.group_size)
            quantized_parts.append(
                self._quantize_groups(full_groups).reshape(*values.shape[:-1], full_width)
            )
        if full_width < width:
            tail_group = values[..., full_width:].unsqueeze(-2)
            quantized_parts.append(self._quantize_groups(tail_group).squeeze(-2))
        return torch.cat(quantized_parts, dim=-1).to(dtype=tensor.dtype)

    def _quantize_groups(self, groups: torch.Tensor) -> torch.Tensor:
        if self.int2_mode != "affine":
            from .oscar import oscar_lloyd_int2_qdq

            return oscar_lloyd_int2_qdq(groups, groups.shape[-1])
        minimum = groups.amin(dim=-1, keepdim=True)
        maximum = groups.amax(dim=-1, keepdim=True)
        scale = (maximum - minimum) / self.qmax
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round((groups - minimum) / safe_scale).clamp(self.qmin, self.qmax)
        return codes * scale + minimum

    def quantize_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """QDQ independent one-token rows ``[N, ..., 1, R]``, exact per row.

        Row ``i`` of the result equals ``self(rows[i : i + 1])`` bit for bit.
        The sliding window uses this to quantize, in one go, the rows that
        leave the recent window during a decode step (at most one per batch
        element) without changing the numerics of the historical per-row
        calls.  Affine INT only uses exact min/max reductions and elementwise
        ops, so one batched call is exact per row.  The other INT2 modes reduce
        means/stds whose CUDA reduction order may depend on the batch shape, so
        they keep one call per row.
        """
        if self.int2_mode == "affine":
            return self(rows)
        return torch.cat([self(rows[index : index + 1]) for index in range(rows.shape[0])])

    def __repr__(self) -> str:
        return f"FakeIntQuantizer(bits={self.bits}, group_size={self.group_size})"


class QuaRotFakeQuantizer(FakeIntQuantizer):
    """QuaRot: segmented Hadamard rotation followed by naive affine INT quant.

    OSCAR's QuaRot baseline applies a fixed, data-free FWHT along the head
    dimension, then uses the ordinary per-token/group INT2 quantizer.  The
    rotation is block-diagonal when ``hadamard_order < width`` (the OSCAR
    implementation defaults to order 16), so each contiguous FWHT block is
    rotated independently before the existing groupwise quantization.
    """

    def __init__(
        self,
        bits: int = 2,
        group_size: int = 128,
        hadamard_order: int = 16,
        int2_mode: str = "affine",
    ) -> None:
        super().__init__(bits=bits, group_size=group_size, int2_mode=int2_mode)
        if not isinstance(hadamard_order, int) or hadamard_order < 2:
            raise ValueError("hadamard_order must be an integer >= 2")
        if hadamard_order & hadamard_order - 1:
            raise ValueError(f"hadamard_order must be a power of two, got {hadamard_order}")
        self.hadamard_order = hadamard_order
        self._hadamard: dict[tuple[str, int], torch.Tensor] = {}

    def _get_hadamard(self, tensor: torch.Tensor) -> torch.Tensor:
        key = (str(tensor.device), self.hadamard_order)
        matrix = self._hadamard.get(key)
        if matrix is None:
            matrix = _build_hadamard(self.hadamard_order, tensor.device)
            self._hadamard[key] = matrix
        return matrix

    def _rotate(self, tensor: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        width = tensor.shape[-1]
        if width == 0 or width % self.hadamard_order:
            raise ValueError(
                f"QuaRot latent width must be divisible by hadamard_order, got width={width}, order={self.hadamard_order}"
            )
        hadamard = self._get_hadamard(tensor)
        blocks = tensor.float().reshape(
            *tensor.shape[:-1], width // self.hadamard_order, self.hadamard_order
        )
        if inverse:
            hadamard = hadamard.transpose(0, 1)
        return torch.matmul(blocks, hadamard).reshape_as(tensor)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_floating_point():
            raise TypeError(f"fake quantization expects a floating tensor, got {tensor.dtype}")
        rotated = self._rotate(tensor)
        rotated = rotated.to(dtype=tensor.dtype)
        quantized = super().__call__(rotated)
        return self._rotate(quantized, inverse=True).to(dtype=tensor.dtype)

    def _rotate_rows(self, rows: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """``_rotate`` that keeps the per-row GEMM shape of a per-row call.

        A per-row call ``self(rows[i : i + 1])`` rotates that row's
        ``R / order`` blocks with one matmul of ``M = R / order`` rows.  One
        matmul over all rows (``M = N * R / order``) can select a different
        cuBLAS kernel and round differently (measured on A100: it does for
        most ``N >= 5``).  ``bmm`` with one problem per row keeps ``M`` per
        problem and reproduces the per-row result bit for bit.
        """
        width = rows.shape[-1]
        if width == 0 or width % self.hadamard_order:
            raise ValueError(
                f"QuaRot latent width must be divisible by hadamard_order, got width={width}, order={self.hadamard_order}"
            )
        hadamard = self._get_hadamard(rows)
        if inverse:
            hadamard = hadamard.transpose(0, 1)
        count = rows.shape[0]
        blocks = rows.float().reshape(count, -1, self.hadamard_order)
        rotated = torch.bmm(blocks, hadamard.expand(count, *hadamard.shape))
        return rotated.reshape(rows.shape)

    def quantize_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """QDQ independent one-token rows, exact per row (see the base class)."""
        if not rows.is_floating_point():
            raise TypeError(f"fake quantization expects a floating tensor, got {rows.dtype}")
        if rows.shape[0] == 0:
            return rows.clone()
        if self.int2_mode != "affine":
            return torch.cat([self(rows[index : index + 1]) for index in range(rows.shape[0])])
        rotated = self._rotate_rows(rows).to(dtype=rows.dtype)
        quantized = FakeIntQuantizer.__call__(self, rotated)
        return self._rotate_rows(quantized, inverse=True).to(dtype=rows.dtype)

    def __repr__(self) -> str:
        return f"QuaRotFakeQuantizer(bits={self.bits}, group_size={self.group_size}, hadamard_order={self.hadamard_order})"


def _build_hadamard(size: int, device: torch.device) -> torch.Tensor:
    if size < 1 or size & size - 1:
        raise ValueError(f"Hadamard size must be a power of two, got {size}")
    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)], dim=0
        )
    return matrix / size**0.5


def _build_block_hadamard(width: int, order: int, device: torch.device) -> torch.Tensor:
    """Block-diagonal ``blockdiag(H_order, ..., H_order)`` of shape ``[width, width]``.

    ``order == width`` reduces to the dense full-width Hadamard, so the KVarN
    H512 path (rotate all latent channels together) and the H128 path (rotate
    each 128-wide quant group independently, matching QuaRot/OSCAR's block
    frame) share one builder. Rotating with a block-diagonal matrix mixes only
    within each ``order``-sized block, which is exactly ``blockdiag(H128 x 4)``
    on the 512-d MLA latent.
    """
    if order < 1 or width % order:
        raise ValueError(
            f"hadamard_order must be a positive divisor of width, got order={order}, width={width}"
        )
    block = _build_hadamard(order, device)
    if order == width:
        return block
    return torch.block_diag(*[block for _ in range(width // order)])


def _kvarn_variance_normalize(
    tiles: torch.Tensor, iterations: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched KVarN Sinkhorn balancing for ``[N, rows, cols]`` tiles.

    The returned scales satisfy ``balanced = tiles / s_col / s_row``. This is
    the same best-so-far log-domain algorithm used by the KVarN MLA reference.
    ``s_col`` varies over columns (tokens in the transposed KVarN layout) and
    ``s_row`` varies over rows (latent channels).
    """
    values = tiles.float()
    n_tiles, rows, cols = values.shape
    device = values.device
    log_col = torch.zeros((n_tiles, 1, cols), device=device)
    log_row = torch.zeros((n_tiles, rows, 1), device=device)

    def imbalance(current: torch.Tensor) -> torch.Tensor:
        col_std = current.std(dim=1)
        row_std = current.std(dim=2)
        return col_std.amax(dim=-1) / col_std.amin(dim=-1).clamp_min(1e-08) + row_std.amax(
            dim=-1
        ) / row_std.amin(dim=-1).clamp_min(1e-08)

    current = values
    best_score = imbalance(current)
    best_col = torch.ones_like(log_col)
    best_row = torch.ones_like(log_row)
    for _ in range(iterations):
        col_std = current.std(dim=1, keepdim=True).clamp(0.001, 1000.0)
        log_col = (log_col + col_std.log()).clamp(-0.3, 10.0)
        current = values / log_col.exp() / log_row.exp()
        row_std = current.std(dim=2, keepdim=True).clamp(0.001, 1000.0)
        log_row = (log_row + row_std.log()).clamp(-0.3, 10.0)
        current = values / log_col.exp() / log_row.exp()
        score = imbalance(current)
        better = score <= best_score
        mask = better[:, None, None]
        best_col = torch.where(mask, log_col.exp(), best_col)
        best_row = torch.where(mask, log_row.exp(), best_row)
        best_score = torch.where(better, score, best_score)
    return (values / best_col / best_row, best_col, best_row)


class KVarNFakeQuantizer(FakeIntQuantizer):
    """KVarN Hadamard + Sinkhorn + tile-wise affine fake quantizer.

    MLA's ``k_nope`` is laid out as ``[batch, 1, tokens, kv_lora_rank]``.
    KVarN transposes each token tile to ``[rank, group_tokens]``, balances
    that matrix, quantizes each latent channel across the tile, then restores
    the two orthogonal transforms. The result is still the original floating
    dtype, so this class measures numerical quality rather than packed memory.
    """

    def __init__(
        self,
        bits: int = 2,
        group_size: int = 128,
        sinkhorn_iterations: int = 16,
        partial: str = "tile",
        hadamard_order: int | None = None,
    ) -> None:
        super().__init__(bits=bits, group_size=group_size)
        if partial not in {"tile", "per-token"}:
            raise ValueError(f"partial must be 'tile' or 'per-token', got {partial!r}")
        self.sinkhorn_iterations = sinkhorn_iterations
        self.partial = partial
        if hadamard_order is not None:
            if not isinstance(hadamard_order, int) or hadamard_order < 2:
                raise ValueError("hadamard_order must be an integer >= 2")
            if hadamard_order & hadamard_order - 1:
                raise ValueError(f"hadamard_order must be a power of two, got {hadamard_order}")
        self.hadamard_order = hadamard_order
        self._hadamard: dict[tuple[str, int], torch.Tensor] = {}

    def _get_hadamard(self, width: int, tensor: torch.Tensor) -> torch.Tensor:
        order = self.hadamard_order or width
        if width % order:
            raise ValueError(
                f"KVarN latent width {width} must be divisible by hadamard_order {order}"
            )
        key = (str(tensor.device), width, order)
        matrix = self._hadamard.get(key)
        if matrix is None:
            matrix = _build_block_hadamard(width, order, tensor.device)
            self._hadamard[key] = matrix
        return matrix

    def _quantize_tile(self, tile: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
        """Quantize one ``[..., tokens, rank]`` tile and return its dtype."""
        tokens, width = tile.shape[-2:]
        flat = tile.reshape(-1, tokens, width).float()
        if tokens < 2:
            rotated = torch.matmul(flat, hadamard)
            groups = rotated.reshape(*rotated.shape[:-1], -1, self.group_size)
            minimum = groups.amin(dim=-1, keepdim=True)
            maximum = groups.amax(dim=-1, keepdim=True)
            scale = (maximum - minimum) / self.qmax
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            codes = torch.round((groups - minimum) / safe_scale).clamp(self.qmin, self.qmax)
            restored = (codes * scale + minimum).reshape_as(rotated)
            return (
                torch.matmul(restored, hadamard.transpose(0, 1)).reshape(tile.shape).to(tile.dtype)
            )
        rotated = torch.matmul(flat, hadamard).transpose(1, 2).contiguous()
        sinkhorn_iterations = self.sinkhorn_iterations
        if os.environ.get("KVARN_SKIP_SINKHORN"):
            sinkhorn_iterations = 0
        balanced, s_col, s_row = _kvarn_variance_normalize(rotated, iterations=sinkhorn_iterations)
        minimum = balanced.amin(dim=-1, keepdim=True)
        maximum = balanced.amax(dim=-1, keepdim=True)
        scale = (maximum - minimum) / self.qmax
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round((balanced - minimum) / safe_scale).clamp(self.qmin, self.qmax)
        dequant_rotated = (codes * (safe_scale * s_row) + minimum * s_row) * s_col
        restored = torch.matmul(dequant_rotated.transpose(1, 2), hadamard.transpose(0, 1)).reshape(
            flat.shape
        )
        return restored.reshape(tile.shape).to(tile.dtype)

    def quantize_full_tiles(
        self, tensor: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        if tensor.ndim < 2 or not tensor.is_floating_point():
            raise TypeError("KVarN expects a floating tensor with token and latent dimensions")
        tokens, width = tensor.shape[-2:]
        if tokens == 0:
            return tensor.clone()
        if width == 0 or width % self.group_size:
            raise ValueError(
                f"KVarN latent width must be divisible by group_size, got {width} and {self.group_size}"
            )
        self.call_count += 1
        self.element_count += tensor.numel()
        if len(self.first_shapes) < 4:
            self.first_shapes.append(tuple(tensor.shape))
        hadamard = self._get_hadamard(width, tensor)
        if valid is not None:
            valid = valid.to(device=tensor.device, dtype=torch.bool)
            if valid.dim() != 2 or valid.shape[0] != tensor.shape[0] or valid.shape[1] != tokens:
                valid = None
        if valid is not None and (not bool(valid.all())):
            return self._quantize_masked(tensor, hadamard, valid)
        return self._quantize_plain(tensor, hadamard)

    def _quantize_masked(
        self, tensor: torch.Tensor, hadamard: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Re-quantize only the chunks where padding shares a chunk with real rows.

        Padding must not reach another row's statistics: a chunk that mixes pad
        and real rows would compute its per-channel min/max (and Sinkhorn
        scaling) over both. Every other chunk is either entirely real or
        entirely padding, so the batched result is already correct for it.
        Rewriting just the mixed chunks keeps the cost near the unmasked path —
        recomputing every chunk per batch element is ~20x slower on padded
        loglikelihood batches.

        Inside a mixed chunk the real rows are moved to the front, quantized in
        their original order (which reproduces exactly the chunking an unpadded
        batch element gets), and moved back; the pad rows of that chunk are
        quantized as their own block so every cached row stays INT2.
        """
        output = self._quantize_plain(tensor, hadamard)
        chunk = self.group_size
        tokens = tensor.shape[-2]
        n_chunks = (tokens + chunk - 1) // chunk
        padded = torch.nn.functional.pad(valid, (0, n_chunks * chunk - tokens), value=False)
        counts = padded.reshape(valid.shape[0], n_chunks, chunk).sum(dim=-1).tolist()
        for index in range(tensor.shape[0]):
            row_valid = valid[index]
            row_counts = counts[index]
            if sum(row_counts) == tokens:
                continue
            for chunk_index, start in enumerate(range(0, tokens, chunk)):
                end = min(start + chunk, tokens)
                if row_counts[chunk_index] in (0, end - start):
                    continue
                segment_valid = row_valid[start:end]
                keep = segment_valid.nonzero(as_tuple=True)[0]
                drop = (~segment_valid).nonzero(as_tuple=True)[0]
                segment = tensor[index : index + 1, ..., start:end, :]
                order = torch.cat([keep, drop])
                permuted = segment.index_select(-2, order)
                pieces = []
                if keep.numel():
                    pieces.append(self._quantize_plain(permuted[..., : keep.numel(), :], hadamard))
                if drop.numel():
                    pieces.append(self._quantize_plain(permuted[..., keep.numel() :, :], hadamard))
                quantized = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)
                inverse = torch.empty_like(order)
                inverse[order] = torch.arange(order.numel(), device=order.device)
                output[index : index + 1, ..., start:end, :] = quantized.index_select(-2, inverse)
        return output

    def _quantize_plain(self, tensor: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
        """The historical behaviour: chunk every row, padding included."""
        tokens = tensor.shape[-2]
        if os.environ.get("KVARN_FORCE_PER_TOKEN"):
            return torch.cat(
                [
                    self._quantize_tile(tensor[..., index : index + 1, :], hadamard)
                    for index in range(tokens)
                ],
                dim=-2,
            )
        chunk = self.group_size
        n_full = tokens // chunk
        chunks = []
        if n_full:
            chunks.append(
                self._quantize_full_chunks(tensor[..., : n_full * chunk, :], hadamard, n_full)
            )
        for start in range(n_full * chunk, tokens, chunk):
            end = min(start + chunk, tokens)
            if (
                end - start < self.group_size
                and self.partial == "per-token"
                and (not os.environ.get("KVARN_SKIP_PARTIAL_TILES"))
            ):
                chunks.append(
                    torch.cat(
                        [
                            self._quantize_tile(tensor[..., index : index + 1, :], hadamard)
                            for index in range(start, end)
                        ],
                        dim=-2,
                    )
                )
                continue
            if end - start < self.group_size and os.environ.get("KVARN_SKIP_PARTIAL_TILES"):
                chunks.append(tensor[..., start:end, :].clone())
            else:
                chunks.append(self._quantize_tile(tensor[..., start:end, :], hadamard))
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-2)

    _MAX_TILES_PER_CALL = 512

    def _quantize_full_chunks(
        self, tensor: torch.Tensor, hadamard: torch.Tensor, n_full: int
    ) -> torch.Tensor:
        """``_quantize_tile`` on every full ``group_size`` chunk, batched.

        Bitwise equal to calling ``_quantize_tile`` once per chunk (the
        historical loop).  The two rotation matmuls stay one call per chunk
        with the same ``M = rows * group_size`` as before, because a matmul
        over all chunks would pick another cuBLAS kernel and round
        differently.  Sinkhorn, min/max and the affine QDQ are per tile, so
        they run once over all tiles.  At 32k tokens this replaces ~256
        Sinkhorn launches per layer with one.
        """
        width = tensor.shape[-1]
        chunk = self.group_size
        lead = tensor.shape[:-2]
        rows = tensor[..., :1, :1].numel()
        sinkhorn_iterations = self.sinkhorn_iterations
        if os.environ.get("KVARN_SKIP_SINKHORN"):
            sinkhorn_iterations = 0
        inverse = hadamard.transpose(0, 1)
        chunks_per_call = max(1, self._MAX_TILES_PER_CALL // max(rows, 1))
        restored_parts = []
        for first in range(0, n_full, chunks_per_call):
            last = min(first + chunks_per_call, n_full)
            rotated = torch.stack(
                [
                    torch.matmul(
                        tensor[..., index * chunk : (index + 1) * chunk, :]
                        .reshape(-1, chunk, width)
                        .float(),
                        hadamard,
                    )
                    .transpose(1, 2)
                    .contiguous()
                    for index in range(first, last)
                ]
            )
            count = last - first
            balanced, s_col, s_row = _kvarn_variance_normalize(
                rotated.reshape(count * rows, width, chunk), iterations=sinkhorn_iterations
            )
            minimum = balanced.amin(dim=-1, keepdim=True)
            maximum = balanced.amax(dim=-1, keepdim=True)
            scale = (maximum - minimum) / self.qmax
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            codes = torch.round((balanced - minimum) / safe_scale).clamp(self.qmin, self.qmax)
            dequant = ((codes * (safe_scale * s_row) + minimum * s_row) * s_col).reshape(
                count, rows, width, chunk
            )
            restored_parts.append(
                torch.stack(
                    [
                        torch.matmul(dequant[index].transpose(1, 2), inverse).reshape(
                            rows, chunk, width
                        )
                        for index in range(count)
                    ],
                    dim=1,
                )
            )
        restored = (
            restored_parts[0] if len(restored_parts) == 1 else torch.cat(restored_parts, dim=1)
        )
        return restored.reshape(*lead, n_full * chunk, width).to(tensor.dtype)

    def __call__(self, tensor: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if os.environ.get("KVARN_SKIP_DECODE_QUANT") and tensor.shape[-2] == 1:
            return tensor.clone()
        return self.quantize_full_tiles(tensor, valid=valid)

    _DIAGNOSTIC_ENV_FLAGS = (
        "KVARN_SKIP_DECODE_QUANT",
        "KVARN_FORCE_PER_TOKEN",
        "KVARN_SKIP_PARTIAL_TILES",
        "KVARN_SKIP_SINKHORN",
    )

    def quantize_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """QDQ independent one-token rows ``[N, ..., 1, R]``, exact per row.

        Row ``i`` equals ``self(rows[i : i + 1], valid=<all true>)``: the
        one-token Hadamard + affine fallback of ``_quantize_tile``.  Its M=1
        rotation matmul dispatches to a cuBLAS GEMV, and neither a batched mm
        (M=N) nor ``bmm`` reproduces that rounding (measured on A100), so the
        two rotations stay one matmul per row.  The exact min/max reductions
        and the elementwise QDQ run once over all rows, and nothing here
        synchronises with the host.
        """
        if rows.ndim < 2 or not rows.is_floating_point():
            raise TypeError("KVarN expects a floating tensor with token and latent dimensions")
        count, tokens, width = (rows.shape[0], rows.shape[-2], rows.shape[-1])
        if count == 0:
            return rows.clone()
        if tokens != 1 or any((os.environ.get(flag) for flag in self._DIAGNOSTIC_ENV_FLAGS)):
            return torch.cat(
                [
                    self(
                        rows[index : index + 1],
                        valid=torch.ones(1, tokens, dtype=torch.bool, device=rows.device),
                    )
                    for index in range(count)
                ]
            )
        if width == 0 or width % self.group_size:
            raise ValueError(
                f"KVarN latent width must be divisible by group_size, got {width} and {self.group_size}"
            )
        self.call_count += count
        self.element_count += rows.numel()
        for _ in range(min(count, 4 - len(self.first_shapes))):
            self.first_shapes.append((1, *rows.shape[1:]))
        hadamard = self._get_hadamard(width, rows)
        heads = rows[0].numel() // width
        flat = rows.reshape(count * heads, width).float()
        rotated = torch.cat(
            [
                torch.mm(flat[index * heads : (index + 1) * heads], hadamard)
                for index in range(count)
            ]
        ).reshape(count, heads, 1, width)
        groups = rotated.reshape(*rotated.shape[:-1], -1, self.group_size)
        minimum = groups.amin(dim=-1, keepdim=True)
        maximum = groups.amax(dim=-1, keepdim=True)
        scale = (maximum - minimum) / self.qmax
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round((groups - minimum) / safe_scale).clamp(self.qmin, self.qmax)
        restored = (codes * scale + minimum).reshape(count * heads, width)
        inverse = hadamard.transpose(0, 1)
        output = torch.cat(
            [
                torch.mm(restored[index * heads : (index + 1) * heads], inverse)
                for index in range(count)
            ]
        )
        return output.reshape(rows.shape).to(rows.dtype)

    def __repr__(self) -> str:
        return f"KVarNFakeQuantizer(bits={self.bits}, group_size={self.group_size}, sinkhorn_iterations={self.sinkhorn_iterations}, partial={self.partial!r}, hadamard_order={self.hadamard_order})"


class FakeQuantizedDynamicCache(DynamicCache):
    """DynamicCache that fake-quantizes selected newly appended MLA latents."""

    def __init__(
        self,
        ddp_cache_data: Any | None = None,
        *,
        config: Any | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
        quantizer: FakeIntQuantizer | OscarMLAQuantizer | None = None,
        components: Iterable[str] = ("nope", "rope"),
        sink_tokens: int = 0,
        recent_tokens: int = 0,
    ) -> None:
        if not isinstance(sink_tokens, int) or isinstance(sink_tokens, bool) or sink_tokens < 0:
            raise ValueError("sink_tokens must be a non-negative integer")
        if (
            not isinstance(recent_tokens, int)
            or isinstance(recent_tokens, bool)
            or recent_tokens < 0
        ):
            raise ValueError("recent_tokens must be a non-negative integer")
        super().__init__(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )
        self.quantizer = quantizer or FakeIntQuantizer()
        self.components = _normalize_components(components)
        self.sink_tokens = sink_tokens
        self.recent_tokens = recent_tokens
        self._row_state: dict[int, torch.Tensor] = {}
        self._target_state_memo: tuple[tuple[int, int, str], torch.Tensor] | None = None
        self._valid_mask_memo: tuple[tuple[int, int, str], torch.Tensor] | None = None
        self._pad_mask: torch.Tensor | None = None
        self._row_cache_key: tuple[int, int] | None = None
        self._row_cache: torch.Tensor | None = None
        if _is_layer_quantizer(self.quantizer):
            if self.components != {"nope"}:
                raise ValueError("OSCAR-MLA requires components=('nope',)")
            if (sink_tokens or recent_tokens) and getattr(
                self.quantizer, "high_precision_mask", None
            ) is not None:
                raise ValueError(
                    "sink/recent window owns high-precision rows; do not set quantizer.high_precision_mask at the same time"
                )
            self.quantizer.validate_config(config)

    def set_pad_mask(self, mask: torch.Tensor | None) -> None:
        """Tell the cache which rows of the next update are padding."""
        if not pad_fix_enabled():
            mask = None
        self._pad_mask = None if mask is None else mask.to("cpu", torch.bool)
        self._row_cache_key = None
        self._row_cache = None
        self._target_state_memo = None
        self._valid_mask_memo = None
        if self._pad_mask is not None:
            _PAD_MASK_STATS["forwards_with_candidate_mask"] += 1

    def _valid_rows(self, tokens: int, batch: int, device: torch.device) -> torch.Tensor | None:
        if self._pad_mask is None or self._pad_mask.dim() != 2:
            return None
        if self._pad_mask.shape[0] != batch or self._pad_mask.shape[1] < tokens:
            return None
        key = (tokens, batch)
        if self._row_cache_key != key:
            sliced = self._pad_mask[:, -tokens:].to(device=device).clone()
            self._row_cache = None if bool(sliced.all()) else sliced
            self._row_cache_key = key
            if self._row_cache is not None:
                _PAD_MASK_STATS["forwards_where_padding_was_excluded"] += 1
        return self._row_cache

    def _window_enabled(self) -> bool:
        return self.sink_tokens > 0 or self.recent_tokens > 0

    def _full_valid_mask(self, full_len: int, batch: int, device: torch.device) -> torch.Tensor:
        key = (full_len, batch, str(device))
        if self._valid_mask_memo is not None and self._valid_mask_memo[0] == key:
            return self._valid_mask_memo[1]
        if self._pad_mask is None:
            valid = torch.ones(batch, full_len, dtype=torch.bool, device=device)
        else:
            mask = self._pad_mask
            if mask.shape[0] != batch or mask.shape[1] < full_len:
                valid = torch.ones(batch, full_len, dtype=torch.bool, device=device)
            else:
                valid = mask[:, -full_len:].to(device=device)
        self._valid_mask_memo = (key, valid)
        return valid

    def _target_window_state(self, full_len: int, batch: int, device: torch.device) -> torch.Tensor:
        """Per-row target state after the current append.

        0 = padding / invalid, 1 = BF16 high precision, 2 = quantized.  Sink
        positions are the first valid tokens of each row, recent positions are
        the last valid tokens.  This is padding-aware, so left-padded
        generation keeps the first *real* tokens as sink, not the pad tokens.
        """
        key = (full_len, batch, str(device))
        if self._target_state_memo is not None and self._target_state_memo[0] == key:
            return self._target_state_memo[1].clone()
        valid = self._full_valid_mask(full_len, batch, device)
        state = valid.to(torch.uint8) * 2
        if self.sink_tokens:
            from_start = valid.to(torch.int64).cumsum(dim=-1)
            sink = valid & (from_start <= self.sink_tokens)
        else:
            sink = torch.zeros_like(valid)
        if self.recent_tokens:
            from_end = valid.flip(-1).to(torch.int64).cumsum(dim=-1).flip(-1)
            recent = valid & (from_end <= self.recent_tokens)
        else:
            recent = torch.zeros_like(valid)
        state.masked_fill_(sink | recent, 1)
        self._target_state_memo = (key, state)
        return state.clone()

    def _quantize_component(
        self,
        tensor: torch.Tensor,
        layer_idx: int,
        component: str,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if component == "nope":
            if _is_layer_quantizer(self.quantizer):
                return self.quantizer.quantize_layer(tensor, layer_idx, "nope")
            if isinstance(self.quantizer, KVarNFakeQuantizer):
                if valid is None:
                    valid = self._valid_rows(tensor.shape[-2], tensor.shape[0], tensor.device)
                return self.quantizer(tensor, valid=valid)
            return self.quantizer(tensor)
        return self.quantizer(tensor)

    def _apply_window_to_new(
        self, tensor: torch.Tensor, layer_idx: int, component: str, quantized_mask: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """Quantize selected rows and return whether QDQ actually ran.

        OSCAR/learned and the plain groupwise quantizer are gathered into one
        call (see ``_is_row_local_quantizer``).  QuaRot keeps one call per
        batch row: its rotation GEMM rounds differently once the rows of all
        batch elements share one matmul, and its window results have always
        come from per-row calls.  KVarN gathers all batch elements into one
        masked call and keeps per-row calls for single-token rows.
        """
        if _is_row_local_quantizer(self.quantizer):
            head_shape = tensor.shape[1:-2]
            expanded = quantized_mask.reshape(
                tensor.shape[0], *[1] * len(head_shape), tensor.shape[-2]
            ).expand(tensor.shape[:-1])
            flat = tensor.reshape(-1, tensor.shape[-1])
            selected = flat[expanded.reshape(-1)]
            if selected.numel() == 0:
                return (tensor, False)
            quantized = self._quantize_component(
                selected.reshape(-1, tensor.shape[-1]), layer_idx, component
            )
            output = flat.clone()
            output[expanded.reshape(-1)] = quantized.reshape(-1, tensor.shape[-1])
            return (output.reshape_as(tensor), True)
        if not isinstance(self.quantizer, QuaRotFakeQuantizer):
            if not bool(quantized_mask.any()):
                return (tensor, False)
            if bool(quantized_mask.all()):
                return (self._quantize_component(tensor, layer_idx, component), True)
            selected, valid, index = self._gather_selected_rows(tensor, quantized_mask)
            if selected is not None:
                quantized = self._quantize_component(selected, layer_idx, component, valid=valid)
                return (self._scatter_selected_rows(tensor, quantized, index, valid), True)
        elif not bool(quantized_mask.any()):
            return (tensor, False)
        output = tensor.clone()
        did_quantize = False
        row_valid = self._valid_rows(tensor.shape[-2], tensor.shape[0], tensor.device)
        for batch_index in range(tensor.shape[0]):
            indices = quantized_mask[batch_index].nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            selected = tensor[batch_index : batch_index + 1].index_select(-2, indices)
            if row_valid is None:
                selected_valid = torch.ones(
                    1, indices.numel(), dtype=torch.bool, device=tensor.device
                )
            else:
                selected_valid = row_valid[batch_index : batch_index + 1].index_select(-1, indices)
            quantized = self._quantize_component(
                selected, layer_idx, component, valid=selected_valid
            )
            output[batch_index : batch_index + 1].index_copy_(-2, indices, quantized)
            did_quantize = True
        return (output, did_quantize)

    def _gather_selected_rows(
        self, tensor: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Left-align every batch element's True positions of ``mask``.

        Returns ``(selected, valid, index)`` with shapes ``[B, ..., M, R]``,
        ``[B, M]`` and ``[B, ..., M, R]`` where ``M`` is the largest per-row
        count.  Padded entries are filled from unselected positions and marked
        invalid, so they never enter a tile's statistics.
        """
        batch, seq = (tensor.shape[0], tensor.shape[-2])
        mask = mask.reshape(mask.shape[0], -1).to(torch.bool)
        if mask.shape[0] != batch:
            return (None, None, None)
        if mask.shape[1] < seq:
            mask = torch.nn.functional.pad(mask, (0, seq - mask.shape[1]), value=False)
        elif mask.shape[1] > seq:
            return (None, None, None)
        max_count = int(mask.sum(-1).max().item())
        if max_count < 2:
            return (None, None, None)
        positions = torch.arange(seq, device=tensor.device).expand(batch, seq)
        order = torch.argsort(mask.logical_not(), dim=-1, stable=True)
        index2d = torch.gather(positions, 1, order[:, :max_count])
        valid = torch.gather(mask, 1, order[:, :max_count])
        shape = tuple(tensor.shape[:-2]) + (max_count, tensor.shape[-1])
        index = index2d.reshape(batch, *[1] * (tensor.ndim - 3), max_count, 1).expand(shape)
        selected = torch.gather(tensor, -2, index)
        return (selected, valid, index)

    def _scatter_selected_rows(
        self,
        tensor: torch.Tensor,
        quantized: torch.Tensor,
        index: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter ``quantized`` back; padded entries go to a dummy slot."""
        batch, seq = (tensor.shape[0], tensor.shape[-2])
        valid_expanded = valid.reshape(
            batch, *[1] * (tensor.ndim - 3), valid.shape[1], 1
        ).expand_as(index)
        dummy = torch.full_like(index, seq)
        target = torch.where(valid_expanded, index, dummy)
        padded = torch.cat([tensor, tensor[..., :1, :]], dim=-2)
        padded.scatter_(-2, target, quantized)
        return padded[..., :seq, :]

    def _requantize_single_rows(
        self, tensor: torch.Tensor, transition: torch.Tensor, quantize_rows: Any
    ) -> torch.Tensor:
        """Decode fast path: every batch row has at most one leaving token.

        The token of each row is gathered with a device-side index (rows with
        no transition gather a dummy token whose QDQ result is discarded), so
        there is no per-row ``nonzero`` host sync.  ``quantize_rows`` is exact
        per row, so the cache matches the historical per-row loop bit for bit.
        """
        batch = tensor.shape[0]
        has = transition.any(dim=-1)
        position = transition.to(torch.int32).argmax(dim=-1)
        index = position.reshape(batch, *[1] * (tensor.ndim - 1)).expand(
            *tensor.shape[:-2], 1, tensor.shape[-1]
        )
        rows = torch.gather(tensor, -2, index)
        quantized = quantize_rows(rows)
        merged = torch.where(has.reshape(batch, *[1] * (tensor.ndim - 1)), quantized, rows)
        output = tensor.clone()
        output.scatter_(-2, index, merged)
        return output

    def _requantize_rows(
        self, tensor: torch.Tensor, layer_idx: int, component: str, transition: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """Re-quantize rows that have just left the recent window."""
        if _is_layer_quantizer(self.quantizer):
            full_transition = torch.cat(
                [
                    transition,
                    torch.zeros(
                        transition.shape[0],
                        tensor.shape[-2] - transition.shape[-1],
                        dtype=torch.bool,
                        device=transition.device,
                    ),
                ],
                dim=-1,
            )
            return self._apply_window_to_new(tensor, layer_idx, component, full_transition)
        most = int(transition.sum(dim=-1).max().item()) if transition.numel() else 0
        if most == 0:
            return (tensor, False)
        quantize_rows = getattr(self.quantizer, "quantize_rows", None)
        if most == 1 and quantize_rows is not None:
            return (self._requantize_single_rows(tensor, transition, quantize_rows), True)
        if _is_row_local_quantizer(self.quantizer):
            full_transition = torch.cat(
                [
                    transition,
                    torch.zeros(
                        transition.shape[0],
                        tensor.shape[-2] - transition.shape[-1],
                        dtype=torch.bool,
                        device=transition.device,
                    ),
                ],
                dim=-1,
            )
            return self._apply_window_to_new(tensor, layer_idx, component, full_transition)
        output = tensor.clone()
        if not isinstance(self.quantizer, QuaRotFakeQuantizer):
            selected, valid, index = self._gather_selected_rows(tensor, transition)
            if selected is not None:
                quantized = self._quantize_component(selected, layer_idx, component, valid=valid)
                return (self._scatter_selected_rows(tensor, quantized, index, valid), True)
        did_quantize = False
        for batch_index in range(tensor.shape[0]):
            indices = transition[batch_index].nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            selected = tensor[batch_index : batch_index + 1].index_select(-2, indices)
            selected_valid = torch.ones(1, indices.numel(), dtype=torch.bool, device=tensor.device)
            quantized = self._quantize_component(
                selected, layer_idx, component, valid=selected_valid
            )
            output[batch_index : batch_index + 1].index_copy_(-2, indices, quantized)
            did_quantize = True
        return (output, did_quantize)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._window_enabled():
            if "nope" in self.components:
                record_layer_call(layer_idx, key_states)
                key_states = self._quantize_component(key_states, layer_idx, "nope")
            if "rope" in self.components:
                value_states = self._quantize_component(value_states, layer_idx, "rope")
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)
        past_len = 0
        if layer_idx < len(self.layers):
            past_len = int(self.layers[layer_idx].get_seq_length())
        new_len = int(key_states.shape[-2])
        full_len = past_len + new_len
        batch = int(key_states.shape[0])
        device = key_states.device
        target = self._target_window_state(full_len, batch, device)
        new_target = target[:, past_len:full_len] == 2
        if "nope" in self.components:
            key_states, did_quantize = self._apply_window_to_new(
                key_states, layer_idx, "nope", new_target
            )
            if did_quantize:
                record_layer_call(layer_idx, key_states)
        if "rope" in self.components:
            value_states, _ = self._apply_window_to_new(value_states, layer_idx, "rope", new_target)
        keys, values = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        old_state = self._row_state.get(layer_idx)
        if old_state is not None and past_len > 0:
            if old_state.shape[0] != batch or old_state.shape[1] < past_len:
                raise RuntimeError(
                    f"sink/recent row state shape does not match the cache: state={tuple(old_state.shape)} batch={batch} past_len={past_len}"
                )
            target[:, :past_len] = torch.maximum(target[:, :past_len], old_state[:, :past_len])
            transition = (old_state[:, :past_len] == 1) & (target[:, :past_len] == 2)
            if _is_layer_quantizer(self.quantizer):
                if "nope" in self.components:
                    keys, _ = self._requantize_rows(keys, layer_idx, "nope", transition)
                if "rope" in self.components:
                    values, _ = self._requantize_rows(values, layer_idx, "rope", transition)
                self.layers[layer_idx].keys = keys
                self.layers[layer_idx].values = values
            else:
                changed = False
                if "nope" in self.components:
                    keys, did = self._requantize_rows(keys, layer_idx, "nope", transition)
                    changed = changed or did
                if "rope" in self.components:
                    values, did = self._requantize_rows(values, layer_idx, "rope", transition)
                    changed = changed or did
                if changed:
                    self.layers[layer_idx].keys = keys
                    self.layers[layer_idx].values = values
            count = transition.sum()
            _pending_window_add("transitions_hp_to_quantized", count)
            _pending_window_add("requantized_rows", count)
        self._row_state[layer_idx] = target.detach()
        _WINDOW_STATS["updates"] += 1
        _pending_window_add("high_precision_rows", (target == 1).sum())
        _pending_window_add("quantized_rows", (target == 2).sum())
        return (keys, values)

    def reset(self) -> None:
        super().reset()
        self._row_state.clear()
        self._pad_mask = None
        self._row_cache_key = None
        self._row_cache = None
        self._target_state_memo = None
        self._valid_mask_memo = None

    def crop(self, tokens_to_remove: int) -> None:
        super().crop(tokens_to_remove)
        self._target_state_memo = None
        self._valid_mask_memo = None
        for layer_idx, state in list(self._row_state.items()):
            current = int(state.shape[-1])
            if tokens_to_remove > 0:
                if tokens_to_remove >= current:
                    continue
                new_len = int(tokens_to_remove)
            else:
                new_len = max(current - abs(int(tokens_to_remove)), 0)
            self._row_state[layer_idx] = state[..., :new_len]

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        self._target_state_memo = None
        self._valid_mask_memo = None
        for layer_idx, state in list(self._row_state.items()):
            self._row_state[layer_idx] = state.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        self._target_state_memo = None
        self._valid_mask_memo = None
        for layer_idx, state in list(self._row_state.items()):
            self._row_state[layer_idx] = state.index_select(0, indices.to(state.device))

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        self._target_state_memo = None
        self._valid_mask_memo = None
        for layer_idx, state in list(self._row_state.items()):
            self._row_state[layer_idx] = state.index_select(0, beam_idx.to(state.device))


def _normalize_components(components: Iterable[str]) -> frozenset[str]:
    normalized = frozenset(components)
    valid = {"nope", "rope"}
    if not normalized or not normalized <= valid:
        raise ValueError(
            f"components must be a non-empty subset of {{'nope', 'rope'}}, got {sorted(normalized)!r}"
        )
    return normalized


def _is_layer_quantizer(quantizer: Any) -> bool:
    """True for quantizers that own a per-layer latent QDQ entry point.

    ``OscarMLAQuantizer`` and ``LearnedRotationQuantizer`` both expose
    ``quantize_layer``/``validate_config``; KVarN and the generic groupwise
    quantizer deliberately do not.
    """
    return isinstance(quantizer, OscarMLAQuantizer) or (
        hasattr(quantizer, "quantize_layer") and hasattr(quantizer, "validate_config")
    )


def _is_row_local_quantizer(quantizer: Any) -> bool:
    """True when the window may gather the selected rows of all batch
    elements into one quantizer call.

    OSCAR/learned quantizers own a per-layer entry point and their window
    results have always come from this batched path.  The plain affine
    groupwise quantizer only uses exact min/max reductions and elementwise
    ops, so batching is exact for it.  ``QuaRotFakeQuantizer`` is row-local in
    exact arithmetic, but its Hadamard rotation is an FP32 GEMM whose cuBLAS
    kernel (and rounding) changes with the number of rows M ([M,128]@[128,128]
    on A100 switches kernels near M=3072 and M=6144), so batching changes its
    numerics; it keeps per-row calls.  KVarN's tile statistics span tokens.
    """
    if _is_layer_quantizer(quantizer):
        return True
    if isinstance(quantizer, (KVarNFakeQuantizer, QuaRotFakeQuantizer)):
        return False
    return isinstance(quantizer, FakeIntQuantizer)


def _positional_parameter_names(signature: inspect.Signature) -> list[str]:
    return [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]


def _get_call_argument(
    positional_names: list[str], name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if name in positional_names:
        index = positional_names.index(name)
        if index < len(args):
            return args[index]
    return None


def _set_call_argument(
    positional_names: list[str],
    name: str,
    value: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if name in positional_names:
        index = positional_names.index(name)
        if index < len(args):
            mutable_args = list(args)
            mutable_args[index] = value
            return (tuple(mutable_args), kwargs)
    kwargs[name] = value
    return (args, kwargs)


def _build_forward_patch(
    original_forward: Callable[..., Any],
    quantizer: FakeIntQuantizer | OscarMLAQuantizer,
    get_config: Callable[[tuple[Any, ...]], Any],
    cache_cls: type[Cache],
    cache_name: str,
    components: frozenset[str],
    sink_tokens: int = 0,
    recent_tokens: int = 0,
) -> Callable[..., Any]:
    signature = inspect.signature(original_forward)
    positional_names = _positional_parameter_names(signature)

    @functools.wraps(original_forward)
    def forward_with_quantized_cache(*args: Any, **kwargs: Any) -> Any:
        config = get_config(args)
        past_key_values = _get_call_argument(positional_names, "past_key_values", args, kwargs)
        use_cache = _get_call_argument(positional_names, "use_cache", args, kwargs)
        cache_enabled = use_cache if use_cache is not None else getattr(config, "use_cache", True)
        input_ids = _get_call_argument(positional_names, "input_ids", args, kwargs)
        input_length = (
            int(input_ids.shape[-1])
            if isinstance(input_ids, torch.Tensor) and input_ids.dim() >= 2
            else None
        )
        pending_mask = take_pending_pad_mask(input_length)
        attention_mask = _get_call_argument(positional_names, "attention_mask", args, kwargs)
        if pending_mask is not None:
            forward_pad_mask = pending_mask
        elif isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
            forward_pad_mask = None if bool(attention_mask.all()) else attention_mask.to(torch.bool)
        else:
            forward_pad_mask = None
        if _is_layer_quantizer(quantizer):
            quantizer.validate_config(config)
            if not cache_enabled:
                raise ValueError(
                    "OSCAR-MLA requires use_cache=True; otherwise quantization is bypassed"
                )
            if (
                isinstance(past_key_values, cache_cls)
                and past_key_values.quantizer is not quantizer
            ):
                raise ValueError("OSCAR-MLA received a cache owned by another quantizer")
        if cache_enabled and (not isinstance(past_key_values, cache_cls)):
            replaceable = past_key_values is None or (
                isinstance(past_key_values, Cache) and past_key_values.get_seq_length() == 0
            )
            if not replaceable:
                raise ValueError(
                    f"The MLA {cache_name} patch requires a new or empty cache; received {type(past_key_values).__name__}."
                )
            quantized_cache = cache_cls(
                config=config,
                offloading=getattr(past_key_values, "offloading", False),
                offload_only_non_sliding=getattr(past_key_values, "only_non_sliding", False),
                quantizer=quantizer,
                **(
                    {
                        "components": components,
                        "sink_tokens": sink_tokens,
                        "recent_tokens": recent_tokens,
                    }
                    if issubclass(cache_cls, FakeQuantizedDynamicCache)
                    else {}
                ),
            )
            if hasattr(quantized_cache, "set_pad_mask"):
                quantized_cache.set_pad_mask(forward_pad_mask)
            args, kwargs = _set_call_argument(
                positional_names, "past_key_values", quantized_cache, args, kwargs
            )
        elif isinstance(past_key_values, cache_cls) and hasattr(past_key_values, "set_pad_mask"):
            past_key_values.set_pad_mask(forward_pad_mask)
        return original_forward(*args, **kwargs)

    return forward_with_quantized_cache


def patch_transformers_mla_kv_cache(
    *,
    method="naive",
    rotation_dir=None,
    bits=2,
    group_size=128,
    hadamard_order=128,
    storage="fake",
    components=("nope",),
    int2_mode="affine",
    model_types=None,
    sink_tokens=0,
    recent_tokens=0,
    kvarn_partial="tile",
    kvarn_hadamard_order=None,
):
    """Install a content-only QDQ cache before loading a native Transformers model."""
    if storage != "fake" or bits != 2 or group_size != 128 or (set(components) != {"nope"}):
        raise ValueError("this release uses INT2/g128 content QDQ with BF16 RoPE")
    if min(sink_tokens, recent_tokens) < 0:
        raise ValueError("sink/recent counts must be non-negative")
    if _ACTIVE_CLASS_QUANTIZERS:
        raise RuntimeError("unpatch the active method before installing another")
    if method == "learned_rotation":
        from .quantizer import LearnedRotationQuantizer

        quantizer = LearnedRotationQuantizer(rotation_dir)
    elif method == "oscar":
        quantizer = OscarMLAQuantizer(rotation_dir, int2_mode=int2_mode)
    elif method == "kvarn":
        quantizer = KVarNFakeQuantizer(
            bits=bits,
            group_size=group_size,
            partial=kvarn_partial,
            hadamard_order=kvarn_hadamard_order,
        )
    elif method == "quarot":
        quantizer = QuaRotFakeQuantizer(
            bits=bits, group_size=group_size, hadamard_order=hadamard_order, int2_mode="affine"
        )
    elif method == "naive":
        quantizer = FakeIntQuantizer(bits=bits, group_size=group_size)
    else:
        raise ValueError(f"unsupported method: {method}")
    requested = tuple(
        model_types
        or (
            (quantizer.manifest["model"]["model_type"],)
            if _is_layer_quantizer(quantizer)
            else SUPPORTED_TRANSFORMERS_MODELS
        )
    )
    classes = []
    for model_type in requested:
        module_name, class_name = SUPPORTED_TRANSFORMERS_MODELS[model_type]
        classes.append((model_type, getattr(importlib.import_module(module_name), class_name)))
    _reset_runtime_stats()
    for model_type, cls in classes:
        if hasattr(cls, _CLASS_PATCH_STATE):
            raise RuntimeError(f"{cls.__name__} is already patched")
        setattr(cls, _CLASS_PATCH_STATE, cls.forward)
        cls.forward = _build_forward_patch(
            cls.forward,
            quantizer,
            get_config=lambda args: args[0].config,
            cache_cls=FakeQuantizedDynamicCache,
            cache_name=method,
            components=frozenset({"nope"}),
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
        )
        _ACTIVE_CLASS_QUANTIZERS[model_type] = quantizer
    return requested


def unpatch_transformers_mla_kv_cache():
    for model_type in list(_ACTIVE_CLASS_QUANTIZERS):
        module_name, class_name = SUPPORTED_TRANSFORMERS_MODELS[model_type]
        cls = getattr(importlib.import_module(module_name), class_name)
        cls.forward = getattr(cls, _CLASS_PATCH_STATE)
        delattr(cls, _CLASS_PATCH_STATE)
    _ACTIVE_CLASS_QUANTIZERS.clear()
    _PENDING_PAD_MASKS.clear()


def get_transformers_mla_kv_quant_stats():
    _flush_window_stats()
    return {
        "layers": dict(_LAYER_CALL_STATS),
        "padding": dict(_PAD_MASK_STATS),
        "window": dict(_WINDOW_STATS),
        "layer_calls": {
            name: q.layer_calls
            for name, q in _ACTIVE_CLASS_QUANTIZERS.items()
            if _is_layer_quantizer(q)
        },
    }


def get_transformers_mla_kv_quant_metadata():
    return {
        name: q.metadata() for name, q in _ACTIVE_CLASS_QUANTIZERS.items() if _is_layer_quantizer(q)
    }
