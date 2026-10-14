"""Tokenized, disjoint calibration data from a local JSONL text corpus."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import torch
from .artifacts import atomic_write_json, digest_json
from .common import resolve_model


@dataclass
class View:
    item: dict
    input_ids: torch.Tensor
    prediction_positions: torch.Tensor

    @property
    def input_tokens(self):
        return self.input_ids.shape[1]

    @property
    def continuation_tokens(self):
        return self.prediction_positions.numel()

    def to(self, device):
        return View(self.item, self.input_ids.to(device), self.prediction_positions.to(device))


def tokenizer_digest(tokenizer):
    return digest_json({"vocab": tokenizer.get_vocab(), "special": tokenizer.special_tokens_map})


def read_data(path, tokenizer=None, tail=0):
    payload = json.loads(Path(path).read_text())
    expected = payload.get("tokenizer_sha256")
    if expected and tokenizer is not None and expected != tokenizer_digest(tokenizer):
        raise ValueError(f"tokenizer mismatch: {path}")
    records = payload["records"]
    views = []
    sources = set()
    for i, row in enumerate(records):
        ids = row["token_ids"]
        if len(ids) < 2 or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("invalid calibration token sequence")
        if row.get("token_sha256") and digest_json(ids) != row["token_sha256"]:
            raise ValueError("calibration token hash mismatch")
        if row.get("source_sha256") in sources:
            raise ValueError("repeated document in calibration split")
        if row.get("source_sha256"):
            sources.add(row["source_sha256"])
        count = len(ids) - 1
        start = max(0, count - tail) if tail else 0
        views.append(
            View(
                {**row, "label": row.get("label", str(i)), "domain": row.get("domain", "corpus")},
                torch.tensor([ids[:-1]]),
                torch.arange(start, count),
            )
        )
    if not views:
        raise ValueError(f"empty calibration split: {path}")
    return views, payload


def batches(views, budget, tokens_per_update, seed):
    result, current, total, used = [], [], 0, 0
    for view in views:
        remaining = budget - used
        if remaining <= 0:
            break
        if view.input_tokens > remaining:
            view = View(
                view.item,
                view.input_ids[:, :remaining],
                view.prediction_positions[view.prediction_positions < remaining],
            )
        if view.input_tokens > tokens_per_update:
            raise ValueError("tokens-per-update must cover the longest training view")
        if current and total + view.input_tokens > tokens_per_update:
            result.append(current)
            current, total = [], 0
        current.append(view)
        total += view.input_tokens
        used += view.input_tokens
    if current:
        result.append(current)
    if used != budget:
        raise ValueError(f"training data contains {used} input tokens; requested {budget}")
    random.Random(seed).shuffle(result)
    return result


def source_records(args):
    if args.input_jsonl:
        with Path(args.input_jsonl).open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        from datasets import load_dataset

        yield from load_dataset(
            args.dataset, split="train", revision=args.dataset_revision, streaming=True
        )


def prepare(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(resolve_model(args.model), trust_remote_code=False)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    splits = {"rotation": [], "validation": [], "train": []}
    seen, used = set(), 0
    for row in source_records(args):
        text = str(row.get("text", "")).strip()
        source = hashlib.sha256(text.encode()).hexdigest()
        if not text or source in seen:
            continue
        if len(splits["rotation"]) < args.rotation_docs:
            split, length = "rotation", args.rotation_length
        elif len(splits["validation"]) < args.validation_docs:
            split, length = "validation", 512
        else:
            split = "train"
            # Approximate 20/40/40 token proportions at lengths 512/2048/4096.
            length = rng.choices((512, 2048, 4096), weights=(4, 2, 1))[0]
            length = min(length, args.input_token_budget - used)
        ids = tokenizer(text, truncation=True, max_length=length + 1, add_special_tokens=True)[
            "input_ids"
        ]
        if len(ids) < length + 1:
            continue
        seen.add(source)
        record = {
            "label": f"{split}_{len(splits[split])}",
            "token_ids": ids,
            "source_sha256": source,
            "token_sha256": digest_json(ids),
            "domain": row.get("domain", row.get("meta", {}).get("pile_set_name", "corpus")),
        }
        splits[split].append(record)
        if split == "train":
            used += len(ids) - 1
            if used >= args.input_token_budget:
                break
    if used < args.input_token_budget or len(splits["validation"]) != args.validation_docs:
        raise ValueError("insufficient distinct long documents for the requested splits")
    for split, records in splits.items():
        atomic_write_json(
            root / f"{split}.json",
            {
                "format_version": 1,
                "split": split,
                "tokenizer_sha256": tokenizer_digest(tokenizer),
                "seed": args.seed,
                "input_token_budget": sum(len(r["token_ids"]) - 1 for r in records),
                "records": records,
            },
        )
    print(json.dumps({split: len(rows) for split, rows in splits.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", help="JSONL with text and optional domain")
    source.add_argument(
        "--dataset", help="Hugging Face text dataset, streamed from its train split"
    )
    parser.add_argument("--dataset-revision", help="pin the dataset commit for reproducibility")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--input-token-budget", type=int, default=1430800)
    parser.add_argument("--rotation-docs", type=int, default=64)
    parser.add_argument("--rotation-length", type=int, default=512)
    parser.add_argument("--validation-docs", type=int, default=32)
    args = parser.parse_args()
    if (
        min(args.input_token_budget, args.rotation_docs, args.rotation_length, args.validation_docs)
        < 1
    ):
        parser.error("budgets and split sizes must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
