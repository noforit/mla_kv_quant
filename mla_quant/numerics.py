"""Explicit FP32 matmul contract shared by calibration and evaluation."""

import os
import torch


def configure_reference_numerics(precision="highest"):
    """Do this BEFORE loading/forwarding the model, in every entry point.

    `high` is retained solely for explicit historical-result reproduction.
    CUDA FP32 matmul precision is separate from the BF16 model/cast sequence.
    """
    if precision not in ("highest", "high"):
        raise ValueError("reference precision must be highest or explicit legacy high")
    override = os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
    if precision == "highest":
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    torch.set_float32_matmul_precision(precision)
    return {
        "fp32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "tf32_override_before": override,
        "tf32_override_effective": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
        "torch_version": torch.__version__,
        "bf16_model_and_rotation_cast_order_unchanged": True,
        "historical_precision_reproduction": precision != "highest",
    }


def validate_artifact_numerics(manifest):
    """New artifacts fail closed; historical ones predate backend metadata."""
    expected = manifest.get("calibration", {}).get("numerical_backend")
    if expected is None:
        return
    actual = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
    wanted = (expected["fp32_matmul_precision"], expected["cuda_matmul_allow_tf32"])
    if actual != wanted:
        raise ValueError(f"artifact numerical backend mismatch: expected {wanted}, got {actual}")
    if wanted[0] == "highest" and os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise ValueError("artifact numerical backend rejects forced TF32 override")
