"""Collect complete matrix cells and aggregate training seeds without filling gaps."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results")
    parser.add_argument("--suite", choices=("main", "ruler"), required=True)
    parser.add_argument("--output", default="results/summary.json")
    args = parser.parse_args()
    rows = {}
    for path in sorted((Path(args.root) / args.suite).glob("*/*/*/*/scores.json")):
        model, arm, window, item = path.relative_to(Path(args.root) / args.suite).parts[:4]
        config = json.loads((path.parent / "run_config.json").read_text())
        if config.get("limit") is not None:
            continue
        data = json.loads(path.read_text())
        if args.suite == "ruler":
            summary = json.loads((path.parent / "summary.json").read_text())
            if not summary["complete_suite"] or summary["partial"]:
                continue
            value = summary["average"]
        else:
            payload = json.loads((path.parent / f"{item}.json").read_text())
            counts = payload.get("n-samples", {})
            if not counts or any(r["original"] != r["effective"] for r in counts.values()):
                continue
            value = data[item]
        setting = arm.rsplit("_s", 1)[0] if "_s" in arm else arm
        rows.setdefault(f"{model}|{setting}|{window}|{item}", []).append(
            {"arm": arm, "value": value}
        )
    result = {
        key: {
            "mean": statistics.mean(r["value"] for r in values),
            "std": statistics.stdev(r["value"] for r in values) if len(values) > 1 else None,
            "n": len(values),
            "seeds": values,
        }
        for key, values in rows.items()
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Collected {len(result)} groups into {out}")


if __name__ == "__main__":
    main()
