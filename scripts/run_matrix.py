"""Run a main-table or RULER matrix, one independent process per GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def jobs(config, suite, output, data_root):
    settings = config["settings"]
    seeds = config["seeds"]
    for name, model in config["models"].items():
        methods = [(method, method, None) for method in config["baselines"]]
        methods += [
            (
                f"{setting}_s{seed}",
                "ours",
                str(Path(config["training_root"]) / name / setting / f"seed{seed}" / "artifact"),
            )
            for setting, seed in itertools.product(settings, seeds)
        ]
        for (label, method, artifact), window in itertools.product(methods, config["windows"]):
            artifact = model["rotation"] if method == "oscar" else artifact
            if artifact and not (Path(artifact) / "manifest.json").is_file():
                raise FileNotFoundError(f"missing artifact: {artifact}")
            dimensions = config["tasks"] if suite == "main" else model["ruler_lengths"]
            for item in dimensions:
                out = output / suite / name / label / window / str(item)
                cmd = [
                    sys.executable,
                    "-m",
                    f"mla_quant.{'evaluate' if suite == 'main' else 'ruler'}",
                    "--model",
                    model["model"],
                    "--method",
                    method,
                    "--window",
                    window,
                    "--output",
                    str(out),
                    "--seed",
                    "1234",
                ]
                if artifact:
                    cmd += ["--artifact", artifact]
                if suite == "main":
                    cmd += ["--tasks", item]
                    if model.get("cmmlu_data"):
                        cmd += ["--cmmlu-data", model["cmmlu_data"]]
                else:
                    cmd += ["--data-root", str(data_root / name / str(item))]
                yield out, cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/main_table.json"))
    parser.add_argument("--suite", choices=("main", "ruler"), required=True)
    parser.add_argument("--output", default="results")
    parser.add_argument("--data-root", default="data/ruler")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    work = list(jobs(config, args.suite, Path(args.output), Path(args.data_root)))
    if args.dry_run:
        for _, command in work:
            print(json.dumps(command))
        return

    def worker(index):
        statuses = []
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus[index]}
        for out, command in work[index :: len(args.gpus)]:
            if (out / "scores.json").exists():
                if args.suite == "main" or (out / "summary.json").exists():
                    statuses.append({"output": str(out), "status": "existing"})
                    continue
            out.parent.mkdir(parents=True, exist_ok=True)
            command = command + (["--resume"] if out.exists() else [])
            log_path = out.parent / (out.name + ".log")
            with log_path.open("a") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            statuses.append({"output": str(out), "returncode": result.returncode})
        return statuses

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        statuses = list(itertools.chain.from_iterable(pool.map(worker, range(len(args.gpus)))))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.suite}_status.json").write_text(json.dumps(statuses, indent=2))
    if any(s.get("returncode", 0) for s in statuses):
        raise SystemExit("some cells failed; inspect the per-cell logs")


if __name__ == "__main__":
    main()
