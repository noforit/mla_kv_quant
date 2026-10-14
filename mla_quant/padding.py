"""Pass lm-eval's valid row lengths to the MLA cache quantizer.

Padding is excluded from KVarN tile statistics and sink/recent token windows.
The adapter preserves lm-eval's model inputs and attention behavior.
Generation already supplies an attention mask.
"""

from __future__ import annotations
import functools
from typing import Any
import torch

_PATCH_STATE: dict[str, Any] | None = None


def install_pad_mask_patch() -> None:
    """Publish per-row valid lengths from lm-eval's batch padding."""
    global _PATCH_STATE
    if _PATCH_STATE is not None:
        return
    import lm_eval.models.huggingface as hf_module
    import lm_eval.models.utils_hf as utils_module
    from . import cache as mla_kv_quant

    original = utils_module.pad_and_concat

    @functools.wraps(original)
    def pad_and_concat_with_lengths(
        max_length: int, tensors: list[torch.Tensor], padding_side: str = "right"
    ) -> torch.Tensor:
        lengths = [int(tensor.numel()) for tensor in tensors]
        result = original(max_length, tensors, padding_side)
        if padding_side in {"right", "left"} and any((length < max_length for length in lengths)):
            mask = torch.zeros((len(lengths), max_length), dtype=torch.bool)
            for index, length in enumerate(lengths):
                length = min(length, max_length)
                if padding_side == "right":
                    mask[index, :length] = True
                else:
                    mask[index, max_length - length :] = True
            mla_kv_quant.set_pending_pad_mask(mask)
        return result

    utils_module.pad_and_concat = pad_and_concat_with_lengths
    hf_module.pad_and_concat = pad_and_concat_with_lengths
    _PATCH_STATE = {"utils_module": utils_module, "hf_module": hf_module, "original": original}


def uninstall_pad_mask_patch() -> None:
    global _PATCH_STATE
    if _PATCH_STATE is None:
        return
    _PATCH_STATE["utils_module"].pad_and_concat = _PATCH_STATE["original"]
    _PATCH_STATE["hf_module"].pad_and_concat = _PATCH_STATE["original"]
    _PATCH_STATE = None
    from . import cache as mla_kv_quant

    mla_kv_quant.take_pending_pad_mask()
