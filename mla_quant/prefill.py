"""Chunked prefill with native generation after the cache has been populated."""

import torch


def _generate_with_chunked_prefill(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    chunk_tokens: int,
) -> torch.Tensor:
    """Fill the KV cache in chunks, then let ``generate`` do the decoding.

    The last prompt token is deliberately left to ``generate`` so that
    ``cache_position`` and the attention mask stay consistent with the
    single-shot path.  ``logits_to_keep=1`` keeps the per-chunk lm_head
    projection from materialising ``[batch, chunk, 102400]`` logits.
    """
    total = int(input_ids.shape[-1])
    cache = None
    for start in range(0, total - 1, chunk_tokens):
        end = min(start + chunk_tokens, total - 1)
        outputs = model(
            input_ids=input_ids[:, start:end],
            attention_mask=attention_mask[:, :end],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            cache_position=torch.arange(start, end, device=input_ids.device),
        )
        cache = getattr(outputs, "past_key_values", None)
        if cache is None:
            cache = getattr(outputs, "cache", None)
        if cache is None:
            raise RuntimeError("chunked prefill did not return a cache")
        del outputs
    eos_token_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    return model.generate(
        input_ids=input_ids[:, total - 1 : total],
        attention_mask=attention_mask,
        past_key_values=cache,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
