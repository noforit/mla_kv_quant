"""Runtime adapters for evaluating local datasets with unmodified lm-eval."""

from __future__ import annotations
import functools
from pathlib import Path
from typing import Any

_ORIGINAL_LOAD_DATASET: Any | None = None


def install_local_cmmlu_patch(data_dir: str | Path) -> Path:
    """Redirect the lm-eval CMMLU dataset loader to local v1.0.1 CSV files."""
    global _ORIGINAL_LOAD_DATASET
    root = Path(data_dir).expanduser().resolve()
    for split in ("dev", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"CMMLU split directory not found: {split_dir}")
    import datasets

    if _ORIGINAL_LOAD_DATASET is not None:
        return root
    original_load_dataset = datasets.load_dataset

    @functools.wraps(original_load_dataset)
    def load_dataset_with_local_cmmlu(
        path: str, name: str | None = None, *args: Any, **kwargs: Any
    ) -> Any:
        if path != "haonan-li/cmmlu":
            return original_load_dataset(path, name, *args, **kwargs)
        if not name:
            raise ValueError("A CMMLU subject name is required")
        data_files = {split: root / split / f"{name}.csv" for split in ("dev", "test")}
        missing = [str(file_path) for file_path in data_files.values() if not file_path.is_file()]
        if missing:
            raise FileNotFoundError(f"CMMLU files not found for subject {name!r}: {missing}")
        if args:
            raise TypeError("Positional dataset options are not supported for local CMMLU")
        requested_split = kwargs.pop("split", None)
        import pandas as pd

        features = datasets.Features(
            {
                field: datasets.Value("string")
                for field in ("Question", "A", "B", "C", "D", "Answer")
            }
        )
        dataset = datasets.DatasetDict(
            {
                split: datasets.Dataset.from_pandas(
                    pd.read_csv(
                        file_path, dtype=str, encoding="utf-8", index_col=0, keep_default_na=False
                    ),
                    features=features,
                    preserve_index=False,
                )
                for split, file_path in data_files.items()
            }
        )
        return dataset[requested_split] if requested_split is not None else dataset

    datasets.load_dataset = load_dataset_with_local_cmmlu
    _ORIGINAL_LOAD_DATASET = original_load_dataset
    return root


def uninstall_local_cmmlu_patch() -> None:
    """Restore the datasets loader patched by install_local_cmmlu_patch."""
    global _ORIGINAL_LOAD_DATASET
    if _ORIGINAL_LOAD_DATASET is None:
        return
    import datasets

    datasets.load_dataset = _ORIGINAL_LOAD_DATASET
    _ORIGINAL_LOAD_DATASET = None


__all__ = [
    "install_local_cmmlu_patch",
    "uninstall_local_cmmlu_patch",
]
