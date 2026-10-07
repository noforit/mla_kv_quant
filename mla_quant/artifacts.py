"""Small local helpers for immutable experiment outputs and fingerprints."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest_json(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def atomic_write_json(path: str | Path, value) -> None:
    """Atomically replace one progress/metadata file inside its owned run directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def reserve_candidate(path: str | Path) -> Path:
    """Claim a fresh candidate directory before expensive training starts.

    Failed runs leave the directory in place. Choose a new run name; never
    silently delete a historical checkpoint or guess whether it was partial.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def freeze_backbone(model):
    """Freeze leaves, not the student graph; adapter gradients still propagate."""
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return model


def local_checkpoint_fingerprint(model_path: str | Path) -> dict:
    """Portable configuration/index hashes and shard sizes; no machine-specific timestamps."""
    from .oscar import checkpoint_identity

    root = Path(model_path).expanduser().resolve()
    identity = checkpoint_identity(root)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        names = [p.name for p in sorted(root.glob("*.safetensors"))]
        if not names:
            names = [p.name for p in sorted(root.glob("pytorch_model*.bin"))]
    identity["weight_files"] = {name: {"size": (root / name).stat().st_size} for name in names}
    identity["weight_identity_policy"] = "config_index_sha256_and_shard_size"
    return identity


def assert_frozen_backbone(model) -> None:
    if any((p.requires_grad or p.grad is not None for p in model.parameters())):
        raise RuntimeError("backbone was not fully frozen or accumulated unused gradients")
