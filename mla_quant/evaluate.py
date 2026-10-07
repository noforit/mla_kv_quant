"""One main-table cell or the six-task suite with a common cache policy."""

import argparse
import json
import os
from pathlib import Path
from .common import METHODS, WINDOWS, resolve_model, install_method, seed_everything
from . import cache
from .padding import install_pad_mask_patch

TASKS = {"cmmlu": 5, "gsm8k": 5, "humaneval": 0, "mmlu": 5, "mbpp": 3, "triviaqa": 5}


def metric(result, task):
    if task in ("cmmlu", "mmlu"):
        return result["groups"][task]["acc,none"] * 100
    values = result["results"][
        (
            task
            if task != "humaneval"
            else next(
                name
                for name in ("humaneval", "humaneval_instruct_norm")
                if name in result["results"]
            )
        )
    ]
    keys = {
        "gsm8k": "exact_match,flexible-extract",
        "humaneval": "pass@1,create_test",
        "mbpp": "pass_at_1,none",
        "triviaqa": "exact_match,remove_whitespace",
    }
    return values[keys[task]] * 100


def evaluate(args):
    from transformers import AutoConfig
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    import lm_eval

    seed_everything(args.seed)
    model_path = resolve_model(args.model)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    chat = args.chat_template or config.model_type == "minicpm3"
    install_method(args.method, config.model_type, args.artifact, args.window)
    install_pad_mask_patch()
    if args.cmmlu_data:
        from .local_datasets import install_local_cmmlu_patch

        install_local_cmmlu_patch(args.cmmlu_data)
    root = Path(args.output)
    atomic = {
        "model": model_path,
        "method": args.method,
        "artifact": args.artifact,
        "window": args.window,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "chat_template": chat,
        "tasks": args.tasks,
    }
    if args.resume:
        if json.loads((root / "run_config.json").read_text()) != atomic:
            raise ValueError("resume requires the same evaluation configuration")
    else:
        root.mkdir(parents=True, exist_ok=False)
        (root / "run_config.json").write_text(json.dumps(atomic, indent=2))
    model_args = (
        f"pretrained={model_path},dtype=bfloat16,use_cache=True,trust_remote_code=False,"
        f"attn_implementation={args.attention_backend}"
    )
    task_root = Path(lm_eval.__file__).parent / "tasks"
    manager = TaskManager(
        include_defaults=False,
        include_path=[str(task_root / name) for name in TASKS]
        + [str(Path(__file__).parent / "tasks")],
    )
    scores = (
        json.loads((root / "scores.json").read_text())
        if args.resume and (root / "scores.json").exists()
        else {}
    )
    for task in args.tasks:
        if task in scores and (root / f"{task}.json").exists():
            continue
        run_task = "humaneval_instruct_norm" if task == "humaneval" and chat else task
        unsafe = task in ("humaneval", "mbpp")
        if unsafe:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        kwargs = {}
        if task in ("gsm8k", "humaneval"):
            kwargs["gen_kwargs"] = (
                f"do_sample=False,max_gen_toks={256 if task == 'gsm8k' else 1024}"
            )
        result = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=[run_task],
            num_fewshot=TASKS[task],
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
            log_samples=True,
            apply_chat_template=chat,
            confirm_run_unsafe_code=unsafe,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
            task_manager=manager,
            **kwargs,
        )
        if result is None:
            raise RuntimeError(f"no result for {task}")
        samples = result.pop("samples", {})
        (root / f"{task}.json").write_text(json.dumps(result, indent=2, default=str))
        for name, rows in samples.items():
            with (root / f"{name}_samples.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, default=str) + "\n")
        scores[task] = metric(result, task)
        (root / "scores.json").write_text(json.dumps(scores, indent=2))
        print(json.dumps({task: scores[task]}), flush=True)
    if len(scores) == len(TASKS):
        scores["average"] = sum(scores.values()) / len(scores)
    (root / "scores.json").write_text(json.dumps(scores, indent=2))
    (root / "cache_stats.json").write_text(
        json.dumps(cache.get_transformers_mla_kv_quant_stats(), indent=2)
    )
    cache.unpatch_transformers_mla_kv_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--artifact", help="OSCAR rotation or learned artifact directory")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--window", choices=tuple(WINDOWS), default="s4r16")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-backend", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--cmmlu-data", help="optional local CMMLU dev/ and test/ CSV directory")
    parser.add_argument("--limit", type=int, help="partial evaluation; omit for the main table")
    parser.add_argument("--resume", action="store_true")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
