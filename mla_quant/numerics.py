"""Explicit FP32 matmul contract shared by calibration and evaluation."""

import os
import torch


def configure_reference_numerics(precision="highest"):
    """Configure full FP32 matmul precision before model loading and inference."""
    if precision != "highest":
        raise ValueError("RoLaQ uses highest FP32 matmul precision")
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
    }


def validate_artifact_numerics(manifest):
    """Verify the numerical backend recorded in a calibration artifact."""
    expected = manifest.get("calibration", {}).get("numerical_backend")
    if expected is None:
        return
    actual = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)
    wanted = (expected["fp32_matmul_precision"], expected["cuda_matmul_allow_tf32"])
    if actual != wanted:
        raise ValueError(f"artifact numerical backend mismatch: expected {wanted}, got {actual}")
    if wanted[0] == "highest" and os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise ValueError("artifact numerical backend rejects forced TF32 override")
