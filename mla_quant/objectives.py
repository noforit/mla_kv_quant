"""Shared frozen-backbone forwards and memory-bounded full-vocabulary KL."""

from __future__ import annotations
import hashlib
import torch
from torch.utils.checkpoint import checkpoint


def backbone_sha256(model):
    """Exact before/after parameter-byte check, one tensor at a time."""
    digest = hashlib.sha256()
    for name, tensor in model.named_parameters():
        digest.update(name.encode())
        work = tensor.detach().contiguous().view(torch.uint8).cpu()
        digest.update(work.numpy().tobytes())
    return digest.hexdigest()


def forward_hidden(model, example, quantize=None, *, teacher=False, capture_attention=True):
    """Run the native MLA attention; hooks replace only its shared latent."""
    handles, attention_outputs = ([], {})
    positions = example.prediction_positions
    try:
        for index, layer in enumerate(model.model.layers):
            if quantize is not None:
                handles.append(
                    layer.self_attn.kv_a_layernorm.register_forward_hook(
                        lambda module, args, output, layer_idx=index: quantize(layer_idx, output)
                    )
                )
            if capture_attention:

                def hook(module, args, output, layer_idx=index):
                    hidden = output[0] if isinstance(output, tuple) else output
                    selected = hidden.index_select(1, positions)
                    attention_outputs[layer_idx] = selected.detach().cpu() if teacher else selected

                handles.append(layer.self_attn.register_forward_hook(hook))
        hidden = model.model(example.input_ids, use_cache=True).last_hidden_state
        hidden = hidden.index_select(1, positions)
        return (hidden.detach().cpu() if teacher else hidden, attention_outputs)
    finally:
        for handle in handles:
            handle.remove()


def projected_kl(lm_head, teacher_hidden, student_hidden, chunk_size=128, logits_scale=1.0):
    """Full-vocabulary teacher KL, checkpointed in token chunks; retain MiniCPM logit scaling."""
    if teacher_hidden.shape != student_hidden.shape or student_hidden.ndim != 3:
        raise ValueError("teacher/student hidden shape mismatch")
    total = student_hidden.shape[1]
    if total < 1 or chunk_size < 1:
        raise ValueError("KL needs nonempty positions and a positive chunk size")
    if not logits_scale or logits_scale <= 0:
        raise ValueError("logits_scale must be positive")

    def chunk_loss(candidate, reference):
        if logits_scale != 1.0:
            candidate = candidate / logits_scale
            reference = reference / logits_scale
        with torch.no_grad():
            reference = reference.to(candidate.device, candidate.dtype)
            log_p = torch.log_softmax(lm_head(reference).float(), dim=-1)
        log_q = torch.log_softmax(lm_head(candidate).float(), dim=-1)
        return (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()

    loss = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        student = student_hidden[:, start:end]
        teacher = teacher_hidden[:, start:end]
        if torch.is_grad_enabled() and student.requires_grad:
            term = checkpoint(
                chunk_loss, student, teacher, use_reentrant=False, preserve_rng_state=False
            )
        else:
            term = chunk_loss(student, teacher)
        loss = loss + term * ((end - start) / total)
    return loss


def mla_output_loss(student, teacher):
    if not teacher or set(student) != set(teacher):
        raise ValueError("MLA output layer sets differ or are empty")
    terms = []
    for layer, target in teacher.items():
        pred = student[layer].float()
        ref = target.to(pred.device, torch.float32)
        terms.append((pred - ref).square().mean() / ref.square().mean().clamp_min(1e-06))
    return torch.stack(terms).mean()
