"""Joint frozen-model calibration: shared rotation and an INT2 decision threshold."""

import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch import nn
from .artifacts import atomic_write_json, freeze_backbone, assert_frozen_backbone
from .common import resolve_model, seed_everything
from .data import read_data, batches
from .geometry import SharedRotation
from .numerics import configure_reference_numerics
from .objectives import forward_hidden, projected_kl, mla_output_loss
from .oscar import OscarMLAQuantizer, LLOYD_MAX_THRESHOLDS, OSCAR_LLOYD_CENTERS, sha256_file
from .quantizer import rotate_quantize, write_learned_artifact, LearnedRotationQuantizer
from .selection import long_views, select_candidate


def signal_response(threshold):
    phi0 = 1 / math.sqrt(2 * math.pi)
    phi_t = torch.exp(-threshold.square() / 2) * phi0
    inner, outer = abs(OSCAR_LLOYD_CENTERS[1]), abs(OSCAR_LLOYD_CENTERS[3])
    return 2 * (inner * (phi0 - phi_t) + outer * phi_t)


def signal_gain(threshold):
    return 1 / signal_response(threshold)


def threshold_upper_bound():
    low, high = 0.5, 1.5
    for _ in range(60):
        mid = (low + high) / 2
        if signal_gain(torch.tensor(mid)).item() <= 1.3 - 1e-5:
            low = mid
        else:
            high = mid
    return low


def validate(model, views, quantize, reference, chunk_size):
    """Teacher-forced continuation KL relative to the initialization."""
    totals = {}
    scale = float(getattr(model.config, "logits_scaling", 1.0) or 1.0)
    with torch.no_grad():
        for view in views:
            view = view.to(next(model.parameters()).device)
            target, _ = forward_hidden(model, view, teacher=True, capture_attention=False)
            for name, callback in (("candidate", quantize), ("reference", reference)):
                hidden, _ = forward_hidden(model, view, callback, capture_attention=False)
                value = projected_kl(
                    model.lm_head, target, hidden, chunk_size, logits_scale=scale
                ).item()
                key = (name, view.item["domain"])
                acc, count = totals.get(key, (0.0, 0))
                totals[key] = (
                    acc + value * view.continuation_tokens,
                    count + view.continuation_tokens,
                )
    domains = {domain for name, domain in totals if name == "reference"}
    result = {
        domain: {
            name: totals[name, domain][0] / totals[name, domain][1]
            for name in ("candidate", "reference")
        }
        for domain in sorted(domains)
    }
    for values in result.values():
        values["ratio"] = values["candidate"] / max(values["reference"], 1e-6)
    return sum(v["ratio"] for v in result.values()) / len(result), result


def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_everything(args.seed)
    model_path = resolve_model(args.model)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    views, _ = read_data(args.train_data, tokenizer)
    validation, _ = read_data(args.validation_data, tokenizer, tail=96)
    long_validation = long_views(validation, tokenizer.eos_token_id)
    training_sources = {v.item.get("source_sha256") for v in views}
    validation_sources = {v.item.get("source_sha256") for v in validation}
    if (training_sources & validation_sources) - {None}:
        raise ValueError("training and validation documents overlap")
    schedule = batches(
        views,
        args.input_token_budget,
        args.tokens_per_update,
        args.batch_order_seed if args.batch_order_seed is not None else args.seed,
    )
    if args.steps is not None:
        schedule = schedule[: args.steps]
    if not schedule:
        raise ValueError("empty training schedule")
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=getattr(torch, args.dtype),
            trust_remote_code=False,
            attn_implementation=args.attention_backend,
        )
        .to(args.device)
        .eval()
    )
    freeze_backbone(model)
    reference = OscarMLAQuantizer(args.rotation_dir, int2_mode="lloyd_up")
    reference.validate_config(model.config)
    rotations = nn.ModuleList(
        [
            SharedRotation(reference.rotations[i].to(args.device), layer.self_attn)
            for i, layer in enumerate(model.model.layers)
        ]
    )
    count = len(rotations)
    threshold = nn.Parameter(torch.full((count,), LLOYD_MAX_THRESHOLDS[-1], device=args.device))
    quant_params = [threshold]
    optimizer = torch.optim.Adam(
        [
            {"params": rotations.parameters(), "lr": args.lr_rotation},
            {"params": quant_params, "lr": args.lr_quantizer},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(schedule))
    upper = threshold_upper_bound()
    scale = float(getattr(model.config, "logits_scaling", 1.0) or 1.0)
    numerics = configure_reference_numerics()
    config = {
        **vars(args),
        "model_path": model_path,
        "updates": len(schedule),
        "input_tokens": sum(v.input_tokens for b in schedule for v in b),
        "train_data_sha256": sha256_file(args.train_data),
        "validation_data_sha256": sha256_file(args.validation_data),
        "numerical_backend": numerics,
    }
    atomic_write_json(out / "run_config.json", config)
    gain = lambda: signal_gain(threshold)

    def callback(matrices, temperature=0.1):
        gains = gain()
        return lambda i, x: rotate_quantize(
            x,
            matrices[i],
            128,
            "oscar_lloyd",
            threshold[i],
            gains[i],
            x.new_tensor(temperature, dtype=torch.float32),
        )

    baseline = lambda i, x: reference.quantize_layer(x, i, "nope")
    started, events = time.time(), []
    retained = {}

    def evaluate(step):
        matrices = [r() for r in rotations]
        score, domains = validate(
            model, validation, callback(matrices), baseline, args.kl_chunk_size
        )
        if not math.isfinite(score):
            raise RuntimeError("non-finite validation score")
        record = {"step": step, "validation_ratio": score, "domains": domains}
        events.append(record)
        atomic_write_json(out / "validation.json", events)
        retained[step] = record
        keep = set(
            sorted(retained, key=lambda key: (retained[key]["validation_ratio"], key))[:2]
        ) | {0}
        if step in keep:
            torch.save(
                {
                    "step": step,
                    "score": score,
                    "rotations": [r.detach().cpu() for r in matrices],
                    "threshold": threshold.detach().cpu(),
                    "gain": gain().detach().cpu(),
                },
                out / f"candidate_{step}.pt",
            )
        for old_step in list(retained):
            if old_step not in keep:
                retained.pop(old_step)
                (out / f"candidate_{old_step}.pt").unlink(missing_ok=True)
        print(json.dumps(record), flush=True)

    evaluate(0)
    params = list(rotations.parameters()) + quant_params
    with (out / "training.jsonl").open("w") as log:
        for step, batch in enumerate(schedule, 1):
            optimizer.zero_grad(set_to_none=True)
            positions = sum(v.continuation_tokens for v in batch)
            progress = (step - 1) / max(len(schedule) - 1, 1)
            temperature = 0.5 - 0.4 * progress
            losses = {"kl": 0.0, "attention": 0.0}
            for view in batch:
                view = view.to(args.device)
                with torch.no_grad():
                    target, target_attn = forward_hidden(
                        model, view, teacher=True, capture_attention=args.mla_output_weight != 0
                    )
                # Rebuild the rotation graph for each view and backward call.
                matrices = [r() for r in rotations]
                hidden, outputs = forward_hidden(
                    model,
                    view,
                    callback(matrices, temperature),
                    capture_attention=args.mla_output_weight != 0,
                )
                kl = projected_kl(
                    model.lm_head, target, hidden, args.kl_chunk_size, logits_scale=scale
                )
                mla = (
                    mla_output_loss(outputs, target_attn)
                    if args.mla_output_weight
                    else kl.new_zeros(())
                )
                weight = view.continuation_tokens / positions
                loss = (kl + args.mla_output_weight * mla) * weight
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}")
                loss.backward()
                losses["kl"] += kl.item() * weight
                losses["attention"] += mla.item() * weight
                del target, target_attn, matrices, hidden, outputs, loss, kl, mla
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise RuntimeError("missing or non-finite calibration gradients")
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                threshold.clamp_(0.5, upper)
            assert_frozen_backbone(model)
            row = {"step": step, **losses, "elapsed_seconds": time.time() - started}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            if step % args.eval_every == 0 or step == len(schedule):
                evaluate(step)

    candidates = []
    for candidate_step, record in sorted(retained.items()):
        state = torch.load(
            out / f"candidate_{candidate_step}.pt", map_location="cpu", weights_only=True
        )
        matrices = [matrix.to(args.device) for matrix in state["rotations"]]
        frozen_t, frozen_g = state["threshold"].to(args.device), state["gain"].to(args.device)

        def frozen_callback(i, x):
            return rotate_quantize(x, matrices[i], 128, "oscar_lloyd", frozen_t[i], frozen_g[i])

        row = {
            "step": candidate_step,
            "short_ratio": record["validation_ratio"],
            "worst_domain_ratio": max(d["ratio"] for d in record["domains"].values()),
        }
        for length, views in long_validation.items():
            row[f"ratio_{length // 1024}k"] = validate(
                model, views, frozen_callback, baseline, args.kl_chunk_size
            )[0]
        candidates.append(row)
    selected = select_candidate(candidates)
    atomic_write_json(
        out / "selection.json",
        {
            "candidates": candidates,
            "selected": selected,
            "rule": "min S among eligible retained checkpoints; ties prefer fewer steps",
        },
    )
    state = torch.load(
        out / f"candidate_{selected['step']}.pt", map_location="cpu", weights_only=True
    )
    calibration = {
        "seed": args.seed,
        "setting": "kv_energy",
        "signal_coefficient": 1.0,
        "gain_rule": "kappa_over_gaussian_response_C(T)",
        "numerical_backend": numerics,
        "selected_step": state["step"],
        "validation_ratio": state["score"],
        "selection_S": selected["S"],
        "data_sha256": config["train_data_sha256"],
    }
    write_learned_artifact(
        out / "artifact",
        dict(enumerate(state["rotations"])),
        params={i: (float(state["threshold"][i]), float(state["gain"][i])) for i in range(count)},
        model_path=model_path,
        base_rotation_dir=args.rotation_dir,
        family="oscar_lloyd",
        train_part="joint",
        mla_output_weight=args.mla_output_weight,
        calibration=calibration,
    )
    exported = LearnedRotationQuantizer(out / "artifact")
    exported.validate_config(model.config)
    atomic_write_json(
        out / "summary.json",
        {
            "selected_step": state["step"],
            "validation_ratio": state["score"],
            "selection_S": selected["S"],
            "input_tokens": config["input_tokens"],
            "updates": len(schedule),
            "partial": args.steps is not None,
            "elapsed_seconds": time.time() - started,
            "artifact": "artifact",
            "layers": count,
        },
    )
    print(out / "artifact")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "train-data", "validation-data", "rotation-dir", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-order-seed", type=int)
    parser.add_argument("--input-token-budget", type=int, default=1430800)
    parser.add_argument("--tokens-per-update", type=int, default=4096)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--lr-rotation", type=float, default=1e-3)
    parser.add_argument("--lr-quantizer", type=float, default=5e-3)
    parser.add_argument("--mla-output-weight", type=float, default=0.1)
    parser.add_argument("--kl-chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attention-backend", default="flash_attention_2")
    args = parser.parse_args()
    if (
        min(args.input_token_budget, args.tokens_per_update, args.eval_every, args.kl_chunk_size)
        < 1
    ):
        parser.error("token budgets and intervals must be positive")
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    train(args)


if __name__ == "__main__":
    main()
