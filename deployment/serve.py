"""Serve a frozen RoLaQ artifact with native packed INT2 on one CUDA GPU."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_artifact(path, model):
    manifest = json.loads((path / "manifest.json").read_text())
    config = json.loads((model / "config.json").read_text())
    if manifest.get("recipe") != "learned_shared_latent_rotation_quantizer_v2":
        raise ValueError("Expected a learned v2 RoLaQ artifact")
    if (manifest.get("family"), manifest.get("group_size"), manifest.get("components")) != (
        "oscar_lloyd",
        128,
        ["nope"],
    ):
        raise ValueError("Expected content-only INT2 with group size 128")
    if manifest["calibration"].get("signal_coefficient") != 1.0:
        raise ValueError("This release serves the kappa=1 main method")
    identity = manifest["model"]
    for key in ("model_type", "num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim"):
        if identity[key] != config[key]:
            raise ValueError(f"Artifact/model mismatch: {key}")
    for key, name in (
        ("config_sha256", "config.json"),
        ("weight_index_sha256", "model.safetensors.index.json"),
    ):
        if identity.get(key) and sha256(model / name) != identity[key]:
            raise ValueError(f"Artifact belongs to a different checkpoint: {name}")
    # File sizes and content hashes survive a machine-to-machine copy; mtimes do not.
    for name, record in identity.get("weight_files", {}).items():
        if (model / name).stat().st_size != record["size"]:
            raise ValueError(f"Model weight shard size mismatch: {name}")
    if set(manifest["layers"]) != {str(i) for i in range(config["num_hidden_layers"])}:
        raise ValueError("Artifact must contain every decoder layer")
    for layer, record in manifest["layers"].items():
        if record["file"] != f"layer_{int(layer)}.pt":
            raise ValueError("Unexpected rotation filename")
        if sha256(path / record["file"]) != record["sha256"]:
            raise ValueError(f"Rotation checksum mismatch: layer {layer}")
        if not all(math.isfinite(record[key]) and record[key] > 0 for key in ("param0", "param1")):
            raise ValueError("Invalid threshold/gain")
    return manifest


def prepare_tokenizer(model, target):
    """Load the exact tokenizer JSON with the generic fast-tokenizer class."""
    target.mkdir()
    shutil.copy2(model / "tokenizer.json", target / "tokenizer.json")
    config = json.loads((model / "tokenizer_config.json").read_text())
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config.pop("auto_map", None)
    (target / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    special = model / "special_tokens_map.json"
    if special.exists():
        shutil.copy2(special, target / special.name)
    assert sha256(model / "tokenizer.json") == sha256(target / "tokenizer.json")


def build_command(args):
    model = args.model.expanduser().resolve()
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "deepseek_v2" or config.get("kv_lora_rank") != 512:
        raise ValueError("This packed serving entry point supports DeepSeek-V2-Lite")
    if args.context_length > config["max_position_embeddings"]:
        raise ValueError("context-length exceeds the model configuration")
    manifest = None
    artifact = args.artifact.expanduser().resolve() if args.artifact else None
    if args.method == "rolaq":
        if artifact is None:
            raise ValueError("--artifact is required for RoLaQ")
        manifest = validate_artifact(artifact, model)
    spec = importlib.util.find_spec("sglang")
    if spec is None or not (Path(spec.origin).parent / "srt/mem_cache/rolaq_mla.py").is_file():
        raise ValueError("Install the patched serving environment with deployment/install.py")
    out = args.output or Path("runs") / ("serve_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = out.resolve() / "tokenizer"
    prepare_tokenizer(model, tokenizer)
    prefixes = ("SGLANG_OSCAR_", "SGLANG_ROLAQ_", "SGLANG_MIXED_KV_", "SGLANG_BF16_MLA_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
        NVIDIA_TF32_OVERRIDE="0",
        SGLANG_LLOYD_MAX="0",
        SGLANG_ENABLE_MIXED_KV_WINDOWS="0",
        SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS="0",
    )
    if args.method == "rolaq":
        env.update(
            SGLANG_OSCAR_MLA_KV_ROTATION_PATH=str(artifact),
            SGLANG_ROLAQ_MLA_QUANTIZER_PATH=str(artifact),
            SGLANG_OSCAR_MLA_KV_PACKED="1",
            SGLANG_OSCAR_MLA_KV_BITS="2",
            SGLANG_OSCAR_MLA_KV_GROUP_SIZE="128",
            SGLANG_LLOYD_MAX="1",
            SGLANG_MIXED_KV_PREFIX_TOKENS=str(args.sink),
            SGLANG_MIXED_KV_RECENT_TOKENS=str(args.recent),
            SGLANG_ROLAQ_MLA_ABSORB_READ_ROTATIONS="1",
            SGLANG_ROLAQ_MLA_FUSED_DECODE_WRITE="1",
            SGLANG_OSCAR_MLA_PACKED_GF="1",
            SGLANG_OSCAR_MLA_PACKED_GF_BLOCK_N="64",
            SGLANG_OSCAR_MLA_PACKED_WARPS="8",
            SGLANG_OSCAR_MLA_PACKED_GF_STAGES="1",
            SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS="1",
        )
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model),
        "--tokenizer-path",
        str(tokenizer),
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--tp",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--attention-backend",
        "triton",
        "--prefill-attention-backend",
        "triton",
        "--decode-attention-backend",
        "triton",
        "--moe-runner-backend",
        "triton",
        "--disable-radix-cache",
        "--disable-piecewise-cuda-graph",
        "--context-length",
        str(args.context_length),
        "--chunked-prefill-size",
        "2048",
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--max-running-requests",
        "16",
        "--random-seed",
        "1234",
    ]
    if args.method == "rolaq":
        cmd += ["--triton-attention-num-kv-splits", str(args.kv_splits)]
    cmd += ["--disable-cuda-graph"] if args.disable_cuda_graph else ["--cuda-graph-max-bs", "8"]
    (out / "launch.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "command": cmd,
                "artifact_sha256": sha256(artifact / "manifest.json") if manifest else None,
                "tokenizer_sha256": sha256(tokenizer / "tokenizer.json"),
                "environment": {
                    k: v
                    for k, v in env.items()
                    if k.startswith(prefixes)
                    or k in ("SGLANG_LLOYD_MAX", "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS")
                },
            },
            indent=2,
        )
    )
    return cmd, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="local DeepSeek-V2-Lite checkpoint"
    )
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--method", choices=("rolaq", "bf16"), default="rolaq")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--mem-fraction", type=float, default=0.8)
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--recent", type=int, default=16)
    parser.add_argument("--kv-splits", type=int, default=129)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.sink < 0 or args.recent < 0 or args.context_length < 128 or args.kv_splits < 2:
        parser.error("invalid window, context length, or split count")
    if not 0 < args.mem_fraction < 1:
        parser.error("mem-fraction must be between 0 and 1")
    if not args.dry_run:
        pids = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        if pids:
            raise SystemExit(f"GPU {args.gpu} is occupied; select an idle GPU")
    command, env = build_command(args)
    print(json.dumps(command), flush=True)
    if not args.dry_run:
        os.execve(command[0], command, env)


if __name__ == "__main__":
    main()
