"""Fit one OSCAR initialization per MLA layer on a rotation-only split."""

import argparse
from pathlib import Path
import torch
from .common import resolve_model, seed_everything
from .data import read_data
from .oscar import fit_oscar_rotation, write_rotation_artifact, sha256_file


def calibrate(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_everything(args.seed)
    path = resolve_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    views, _ = read_data(args.data, tokenizer)
    model = (
        AutoModelForCausalLM.from_pretrained(
            path,
            dtype=getattr(torch, args.dtype),
            attn_implementation=args.attention_backend,
            trust_remote_code=False,
        )
        .to(args.device)
        .eval()
    )
    captured = {i: [] for i in range(len(model.model.layers))}
    counts = dict.fromkeys(captured, 0)
    handles = []
    for index, layer in enumerate(model.model.layers):

        def hook(_module, _inputs, output, i=index):
            remaining = args.max_tokens - counts[i]
            if remaining > 0:
                rows = output.detach().reshape(-1, output.shape[-1])[:remaining].cpu()
                captured[i].append(rows)
                counts[i] += rows.shape[0]

        handles.append(layer.self_attn.kv_a_layernorm.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            for view in views:
                model.model(view.input_ids.to(args.device), use_cache=True)
                if min(counts.values()) >= args.max_tokens:
                    break
    finally:
        for handle in handles:
            handle.remove()
    rotations = {i: fit_oscar_rotation(torch.cat(rows), 128) for i, rows in captured.items()}
    write_rotation_artifact(
        args.output,
        rotations,
        model_path=path,
        group_size=128,
        calibration={
            "data_sha256": sha256_file(args.data),
            "seed": args.seed,
            "captured_tokens": counts,
            "collection": "prefill_content_latents",
        },
    )
    print(Path(args.output) / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="rotation.json from prepare-data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attention-backend", default="flash_attention_2")
    args = parser.parse_args()
    if args.max_tokens < 2:
        parser.error("max-tokens must be at least two")
    calibrate(args)


if __name__ == "__main__":
    main()
