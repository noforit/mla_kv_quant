# Shared-Latent INT2 Quantization for MLA

Research code for quantizing the **shared content latent** in Multi-head Latent Attention (MLA). The model weights remain frozen; each layer learns one shared orthogonal rotation and an INT2 threshold. The default **κ = 1** rule couples reconstruction gain to the threshold. RoPE keys remain BF16.

## Methods

| Setting | Training option |
|---|---|
| Full joint | `--setting full_joint` |
| K/V aggregate-energy subspaces (`H_K + H_V`) | `--setting kv_energy` |
| K/V signed-contrast subspaces (`H_K − H_V`) | `--setting kv_weight` |
| Cayley | `--setting cayley` |

All four settings use the same training pipeline and one reconstructed latent for both K and V. `--gain-mode unit` is the default; `--gain-mode free` enables the free-gain comparison.

Evaluation supports **BF16, unrotated INT2, H128, KVarN, OSCAR, and our learned method**. The target models are DeepSeek-V2-Lite, Moonlight-16B-A3B, and MiniCPM3-4B.

## Installation

Python 3.10+ and a CUDA GPU are recommended. The package pins the native MLA support in Transformers 5.16.1 and the project's lm-evaluation-harness revision.

```bash
python -m pip install -e .
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
```

`--model` accepts a local checkpoint directory, a Hugging Face ID, or the aliases `deepseek`, `moonlight`, and `minicpm`.

## Prepare data and train

Supply a training-corpus JSONL file with a `text` field. The helper creates disjoint rotation, training, and validation splits.

```bash
mla-prepare-data --model deepseek \
  --input-jsonl /path/to/train.jsonl --output data/deepseek

mla-calibrate --model deepseek --data data/deepseek/rotation.json \
  --output artifacts/oscar/deepseek

CUDA_VISIBLE_DEVICES=0 mla-train --model deepseek \
  --train-data data/deepseek/train.json \
  --validation-data data/deepseek/validation.json \
  --rotation-dir artifacts/oscar/deepseek \
  --setting kv_energy --seed 42 \
  --output runs/train/deepseek/kv_energy/seed42
```

An existing compatible OSCAR rotation directory can replace the calibration step. Training exports a loadable `artifact/` and records its validation-based checkpoint selection.

Train all four settings with seeds 42/43/44:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_all.sh \
  deepseek data/deepseek artifacts/oscar/deepseek runs/train/deepseek
```

## Main-table evaluation

The six tasks are CMMLU, GSM8K, HumanEval, MMLU, MBPP, and TriviaQA.

```bash
mla-evaluate --model deepseek --method ours \
  --artifact runs/train/deepseek/kv_energy/seed42/artifact \
  --window s4r16 --output results/example
```

Use `--method bf16`, `int2`, `h128`, `kvarn`, or `oscar` for baselines. OSCAR needs `--artifact artifacts/oscar/deepseek`. Available protection policies are `nowin`, `s4r8`, and `s4r16`.

Edit model/artifact paths in `configs/main_table.json`, then run the full matrix:

```bash
python scripts/run_matrix.py --suite main --gpus 0 1 --output results
python scripts/collect_results.py --suite main --root results \
  --output results/main_summary.json
```

## RULER

Prepare upstream RULER data for each model tokenizer:

```bash
python -m pip install -e '.[ruler]'
python -m nltk.downloader punkt punkt_tab
python scripts/prepare_ruler.py --model deepseek \
  --output data/ruler/deepseek --lengths 8192 16384
```

For MiniCPM3, use `--template minicpm3-chat`. The generator downloads its external data on the first run.

```bash
mla-ruler --model deepseek --method ours \
  --artifact runs/train/deepseek/kv_energy/seed42/artifact \
  --data-root data/ruler/deepseek/8192 --window s4r16 \
  --output results/ruler_example

python scripts/run_matrix.py --suite ruler --data-root data/ruler \
  --gpus 0 1 --output results
python scripts/collect_results.py --suite ruler --root results \
  --output results/ruler_summary.json
```

Each RULER run covers 13 tasks. Matrix runs use one process per selected GPU and resume existing cells. Partial runs made with `--limit` are excluded from the result collector.

## Code layout

- `mla_quant/geometry.py`: the four shared rotation settings.
- `mla_quant/quantizer.py` and `train.py`: INT2 calibration and artifact export.
- `mla_quant/cache.py` and `oscar.py`: MLA cache integration and baselines.
- `mla_quant/evaluate.py` and `ruler.py`: benchmark entry points.
- `scripts/` and `configs/`: training and evaluation matrices.

This is a quantize–dequantize implementation for accuracy evaluation. Exact reproduction of archived results requires the original calibration data, rotations, and selected artifacts. The portable trainer uses domain-averaged validation KL for checkpoint selection. For MiniCPM3, use the same rotary configuration as the results being reproduced.

Original code is licensed under Apache-2.0. We thank [OSCAR](https://github.com/FutureMLS-Lab/OSCAR), [QuaRot](https://github.com/spcl/QuaRot), [KVarN](https://arxiv.org/abs/2606.03458), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and [RULER](https://github.com/NVIDIA/RULER). Third-party notices are retained in `NOTICE` and `licenses/`.
