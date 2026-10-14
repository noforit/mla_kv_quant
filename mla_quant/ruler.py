"""Evaluate the eight RULER retrieval tasks from upstream-generated JSONL."""

import argparse
import json
from pathlib import Path
import time
import torch
from .common import METHODS, WINDOWS, resolve_model, install_method, seed_everything
from .artifacts import atomic_write_json
from .prefill import _generate_with_chunked_prefill
from .ruler_metrics import string_match_all
from . import cache

MAX_NEW = {
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multiquery": 128,
    "niah_multivalue": 128,
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
}


def generate(model, tokenizer, rows, maximum, prefill_chunk):
    encoded = tokenizer(
        [r["input"] + r.get("answer_prefix", "") for r in rows],
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    encoded = {k: v.to(next(model.parameters()).device) for k, v in encoded.items()}
    length = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        if prefill_chunk and length > prefill_chunk:
            output = _generate_with_chunked_prefill(
                model,
                tokenizer,
                encoded["input_ids"],
                encoded["attention_mask"],
                maximum,
                prefill_chunk,
            )
            continuation = output[:, 1:]
        else:
            output = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=maximum,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=model.generation_config.eos_token_id or tokenizer.eos_token_id,
            )
            continuation = output[:, length:]
    return tokenizer.batch_decode(continuation, skip_special_tokens=True)


def evaluate(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_everything(args.seed)
    path = resolve_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(path, trust_remote_code=False)
    install_method(args.method, config.model_type, args.artifact, args.window)
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
    root = Path(args.output)
    requested = {**vars(args), "model": path}
    if args.resume:
        previous = json.loads((root / "run_config.json").read_text())
        if any(previous.get(k) != requested[k] for k in requested if k != "resume"):
            raise ValueError("resume requires the same model, data, method and generation settings")
    else:
        root.mkdir(parents=True, exist_ok=False)
        atomic_write_json(root / "run_config.json", requested)
    scores = {}
    for task in args.tasks:
        source = Path(args.data_root) / task / "validation.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if args.limit:
            rows = rows[: args.limit]
        if not rows:
            raise ValueError(f"empty RULER task: {source}")
        # Upstream 'index' can repeat; the source row ordinal is the unique ID.
        for ordinal, row in enumerate(rows):
            row["_id"] = ordinal
        output = root / f"{task}.jsonl"
        completed = {}
        if args.resume and output.exists():
            for line in output.read_text().splitlines():
                record = json.loads(line)
                if record["_id"] in completed:
                    raise ValueError("duplicate prediction ID in resume file")
                completed[record["_id"]] = record
        if any(
            i < 0
            or i >= len(rows)
            or r["input"] != rows[i]["input"]
            or r.get("answer_prefix", "") != rows[i].get("answer_prefix", "")
            for i, r in completed.items()
        ):
            raise ValueError("saved predictions do not match the current RULER input")
        pending = sorted(
            [r for r in rows if r["_id"] not in completed],
            key=lambda r: r.get("length", len(r["input"])),
        )
        started = time.time()
        with output.open("a" if args.resume else "w") as handle:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                predictions = generate(
                    model, tokenizer, batch, MAX_NEW[task], args.prefill_chunk_tokens
                )
                for row, prediction in zip(batch, predictions):
                    answers = row.get("outputs", row.get("answers"))
                    if not isinstance(answers, list) or not answers:
                        raise ValueError(f"RULER answers must be a nonempty list: {task}")
                    record = {
                        "_id": row["_id"],
                        "input": row["input"],
                        "answer_prefix": row.get("answer_prefix", ""),
                        "pred": prediction,
                        "answers": answers,
                        "length": row.get("length"),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed[row["_id"]] = record
                handle.flush()
        metric = string_match_all
        ordered = [completed[i] for i in range(len(rows))]
        scores[task] = {
            "score": metric([r["pred"].strip() for r in ordered], [r["answers"] for r in ordered]),
            "samples": len(rows),
            "elapsed_seconds": time.time() - started,
        }
        atomic_write_json(root / "scores.json", scores)
        print(json.dumps({task: scores[task]}), flush=True)
    summary = {
        "suite": "ruler_retrieval_8",
        "tasks": scores,
        "complete_suite": set(scores) == set(MAX_NEW),
        "partial": args.limit is not None,
    }
    if summary["complete_suite"]:
        summary["average"] = sum(r["score"] for r in scores.values()) / len(MAX_NEW)
    atomic_write_json(root / "summary.json", summary)
    atomic_write_json(root / "cache_stats.json", cache.get_transformers_mla_kv_quant_stats())
    cache.unpatch_transformers_mla_kv_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--artifact")
    parser.add_argument(
        "--data-root", required=True, help="one context length: task/validation.jsonl"
    )
    parser.add_argument("--tasks", nargs="+", choices=tuple(MAX_NEW), default=list(MAX_NEW))
    parser.add_argument("--window", choices=tuple(WINDOWS), default="s4r16")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attention-backend", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--limit", type=int, help="partial evaluation; omit for published suite scores"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or (args.limit is not None and args.limit < 1)
        or args.prefill_chunk_tokens < 0
    ):
        parser.error("batch-size and limit must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
