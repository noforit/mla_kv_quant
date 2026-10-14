"""Prepare only the eight NIAH retrieval tasks using the pinned RULER generator."""

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
import yaml
from mla_quant.common import resolve_model
from mla_quant.ruler import MAX_NEW

REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="e.g. data/ruler/deepseek")
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 16384])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--tasks", nargs="+", choices=tuple(MAX_NEW), default=list(MAX_NEW))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default="third_party/RULER")
    parser.add_argument("--template", choices=("base", "minicpm3-chat"), default="base")
    args = parser.parse_args()
    if args.samples < 1 or any(length < 2048 for length in args.lengths):
        parser.error("samples must be positive and retrieval context lengths must be at least 2048")
    source = Path(args.source).resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/NVIDIA/RULER.git", str(source)], check=True
        )
    subprocess.run(["git", "-C", str(source), "checkout", REVISION], check=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = source / "scripts/data"
    tasks = yaml.safe_load((source / "scripts/synthetic.yaml").read_text())
    base = runpy.run_path(str(data / "synthetic/constants.py"))["TASKS"]["niah"]
    templates = runpy.run_path(str(data / "template.py"))["Templates"]
    templates["minicpm3-chat"] = (
        "<|im_start|>user\n{task_template}<|im_end|>\n<|im_start|>assistant\n"
    )
    template = templates[args.template].format(task_template=base["template"])
    template += base.get("answer_prefix", "")
    tokenizer = resolve_model(args.model)
    corpus = data / "synthetic/json"
    if any(tasks[task]["args"]["type_haystack"] == "essay" for task in args.tasks):
        if not (corpus / "PaulGrahamEssays.json").is_file():
            subprocess.run([sys.executable, "download_paulgraham_essay.py"], cwd=corpus, check=True)
        if not json.loads((corpus / "PaulGrahamEssays.json").read_text())["text"].strip():
            raise RuntimeError("The RULER essay background download is empty")
    for length in args.lengths:
        for task in args.tasks:
            task_args = [
                item
                for key, value in tasks[task]["args"].items()
                for item in ("--" + key, str(value))
            ]
            subprocess.run(
                [
                    sys.executable,
                    str(data / "synthetic/niah.py"),
                    "--save_dir",
                    str(output / str(length)),
                    "--save_name",
                    task,
                    "--subset",
                    "validation",
                    "--tokenizer_path",
                    tokenizer,
                    "--tokenizer_type",
                    "hf",
                    "--max_seq_length",
                    str(length),
                    "--num_samples",
                    str(args.samples),
                    "--random_seed",
                    str(args.seed),
                    "--tokens_to_generate",
                    str(base["tokens_to_generate"]),
                    "--template",
                    template,
                    *task_args,
                ],
                cwd=data,
                check=True,
            )
            generated = output / str(length) / task / "validation.jsonl"
            if not generated.is_file() or len(generated.read_text().splitlines()) != args.samples:
                raise RuntimeError(f"upstream generation did not complete: {generated}")
    (output / "preparation.json").write_text(
        json.dumps(
            {
                "ruler_revision": REVISION,
                "model": args.model,
                "tasks": args.tasks,
                "lengths": args.lengths,
                "samples": args.samples,
                "seed": args.seed,
                "template": args.template,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
