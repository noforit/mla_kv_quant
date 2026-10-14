"""Short plus 2K/4K KL selection of the retained calibration checkpoints."""

import torch
from .data import View


def long_views(validation, eos_token_id):
    if len(validation) < 32 or eos_token_id is None:
        raise ValueError("checkpoint selection needs at least 32 validation documents and EOS")
    ordered = sorted(validation, key=lambda view: view.item["label"])[:32]
    views = {2048: [], 4096: []}
    for start in range(0, 32, 8):
        stream = []
        for view in ordered[start : start + 8]:
            if stream:
                stream.append(eos_token_id)
            stream.extend(view.item["token_ids"])
        for length in views:
            tail = stream[-length:]
            if len(tail) != length:
                raise ValueError("validation documents must provide complete 2K/4K views")
            views[length].append(
                View(
                    {"label": f"length_{length}_{start // 8}", "domain": "all"},
                    torch.tensor([tail[:-1]]),
                    torch.arange(len(tail) - 97, len(tail) - 1),
                )
            )
    return views


def select_candidate(candidates):
    """Apply the common short/long quality gates, then choose the smallest S."""
    for row in candidates:
        row["S"] = 0.5 * row["short_ratio"] + 0.25 * (row["ratio_2k"] + row["ratio_4k"])
        row["quality_pass"] = (
            row["S"] <= 1.1
            and row["worst_domain_ratio"] <= 1.25
            and max(row["ratio_2k"], row["ratio_4k"]) <= 1.15
        )
    eligible = [row for row in candidates if row["quality_pass"]] or candidates
    return min(eligible, key=lambda row: (round(row["S"], 6), row["step"]))
