"""Install the pinned SGLang/OSCAR base with the RoLaQ serving patch."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

REVISION = "19f85e13059de3da60686af4cbdd778b5671d9ff"
UPSTREAM = "https://github.com/FutureMLS-Lab/OSCAR.git"
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE.parent / "third_party/OSCAR")
    parser.add_argument("--patch-only", action="store_true", help="skip pip installation")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", UPSTREAM, str(source)],
            check=True,
        )
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", REVISION], check=True)
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != REVISION:
        raise SystemExit(
            f"Expected upstream revision {REVISION}; choose a fresh --source directory"
        )
    command = ["git", "-C", str(source), "apply"]
    patch = str(HERE / "patches/rolaq_sglang.patch")
    check = subprocess.run(command + ["--check", patch], capture_output=True, text=True)
    if check.returncode == 0:
        subprocess.run(command + [patch], check=True)
    elif subprocess.run(command + ["--reverse", "--check", patch], capture_output=True).returncode:
        raise SystemExit("Source does not match the base or the patched version:\n" + check.stderr)
    for module in sorted((HERE / "sglang").rglob("*.py")):
        target = source / "sglang-research/python/sglang" / module.relative_to(HERE / "sglang")
        if target.exists() and target.read_bytes() != module.read_bytes():
            raise SystemExit(f"Serving module has local changes: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(module, target)
    if not args.patch_only:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-c",
                str(HERE / "constraints.txt"),
                "-e",
                str(source / "sglang-research/python"),
            ],
            check=True,
        )
    print(f"RoLaQ serving source ready: {source / 'sglang-research/python'}")


if __name__ == "__main__":
    main()
