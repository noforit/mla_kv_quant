"""Invoke the pinned upstream RULER generator for a model and context lengths."""

import argparse
from pathlib import Path
import subprocess
import sys
from mla_quant.common import resolve_model
from mla_quant.ruler import MAX_NEW

REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="e.g. data/ruler/deepseek")
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 16384])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default="third_party/RULER")
    parser.add_argument("--template", choices=("base", "minicpm3-chat"), default="base")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/NVIDIA/RULER.git", str(source)], check=True
        )
    subprocess.run(["git", "-C", str(source), "checkout", REVISION], check=True)
    template = source / "scripts/data/template.py"
    if args.template == "minicpm3-chat" and "'minicpm3-chat'" not in template.read_text():
        with template.open("a") as handle:
            handle.write(
                "\nTemplates['minicpm3-chat'] = '<|im_start|>user\\n{task_template}<|im_end|>\\n<|im_start|>assistant\\n'\n"
            )
    tokenizer = resolve_model(args.model)
    corpus = source / "scripts/data/synthetic/json"
    if not (corpus / "PaulGrahamEssays.json").is_file():
        subprocess.run([sys.executable, "download_paulgraham_essay.py"], cwd=corpus, check=True)
    if not all((corpus / name).is_file() for name in ("squad.json", "hotpotqa.json")):
        subprocess.run(["bash", "download_qa_dataset.sh"], cwd=corpus, check=True)
    for length in args.lengths:
        for task in MAX_NEW:
            subprocess.run(
                [
                    sys.executable,
                    "prepare.py",
                    "--save_dir",
                    str(Path(args.output).resolve() / str(length)),
                    "--benchmark",
                    "synthetic",
                    "--task",
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
                    "--model_template_type",
                    args.template,
                ],
                cwd=source / "scripts/data",
                check=True,
            )
            generated = Path(args.output).resolve() / str(length) / task / "validation.jsonl"
            if not generated.is_file() or len(generated.read_text().splitlines()) != args.samples:
                raise RuntimeError(f"upstream generation did not complete: {generated}")


if __name__ == "__main__":
    main()
