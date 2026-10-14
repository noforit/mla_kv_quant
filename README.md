# RoLaQ

**Joint Rotation and Range Calibration for 2-Bit KV Caches in Multi-Head Latent Attention**

RoLaQ jointly calibrates a shared rotation and per-layer quantization ranges for
2-bit MLA content caches. It uses joint key/value geometry (`H_K + H_V`) and
combines prediction KL with attention-output loss. Model weights and positional
keys remain BF16.

This repository includes calibration, downstream benchmarks, RULER retrieval,
and SGLang serving with packed INT2 caches.

## Installation

Use Linux, Python 3.12, and a CUDA GPU.
Keep calibration/evaluation and SGLang in separate Conda environments because
they use different PyTorch and Transformers versions.

```bash
git clone https://github.com/noforit/RoLaQ.git
cd RoLaQ
conda create -n rolaq-hf python=3.12 -y
conda activate rolaq-hf
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
```

FlashAttention requires a CUDA toolkit compatible with the installed PyTorch.

Supported models: DeepSeek-V2-Lite, Moonlight-16B-A3B, and MiniCPM3-4B.
Commands accept a local checkpoint, a Hugging Face ID, or the corresponding
alias (`deepseek`, `moonlight`, `minicpm`). The examples below use DeepSeek-V2-Lite.

## Calibration

Download a checkpoint and prepare disjoint initialization, training, and
validation data:

```bash
hf download deepseek-ai/DeepSeek-V2-Lite --local-dir models/deepseek

mla-prepare-data --model models/deepseek \
  --dataset monology/pile-uncopyrighted \
  --dataset-revision 3be90335b66f24456a5d6659d9c8d208c0357119 \
  --output data/deepseek
```

For a local corpus, use `--input-jsonl corpus.jsonl` with a `text` field.

Initialize the shared rotations and jointly calibrate rotation and range:

```bash
CUDA_VISIBLE_DEVICES=0 mla-calibrate --model models/deepseek \
  --data data/deepseek/rotation.json --output artifacts/initial/deepseek

CUDA_VISIBLE_DEVICES=0 mla-train --model models/deepseek \
  --train-data data/deepseek/train.json \
  --validation-data data/deepseek/validation.json \
  --rotation-dir artifacts/initial/deepseek \
  --seed 42 --output runs/deepseek
```

The final artifact is saved to `runs/deepseek/artifact/` and contains each
layer's calibrated rotation and quantizer parameters. Use it for evaluation
and deployment. Data preparation and training require new output directories.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 mla-evaluate --model models/deepseek \
  --method rolaq --artifact runs/deepseek/artifact \
  --window s4r16 --output results/deepseek
```

The suite covers MMLU, CMMLU, TriviaQA, GSM8K, HumanEval, and MBPP with fixed
few-shot settings. Task scores and their mean are saved to `scores.json`.
HumanEval and MBPP run generated Python using lm-eval's code-evaluation backend.

Use `--tasks gsm8k` to select a task, `--batch-size 8` to set the batch size, or
`--resume` to continue an interrupted run. `s4r16` preserves four initial and
sixteen recent tokens in BF16; `--window nowin` disables token protection.
HF evaluation uses quantize–dequantize caches to compare accuracy across methods.

For baselines, set `--method` to `bf16`, `int2`, `h128`, `kvarn`, or `oscar`.
BF16 needs no artifact; OSCAR uses the initialization artifact.

## RULER retrieval

Prepare the data and run RULER retrieval evaluation:

```bash
python -m pip install -e '.[ruler]'
python -m nltk.downloader punkt punkt_tab

python scripts/prepare_ruler.py --model models/deepseek \
  --output data/ruler/deepseek --lengths 16384 --samples 100

CUDA_VISIBLE_DEVICES=0 mla-ruler --model models/deepseek \
  --method rolaq --artifact runs/deepseek/artifact \
  --data-root data/ruler/deepseek/16384 --window s4r16 \
  --batch-size 8 --output results/ruler_deepseek
```

Use `--lengths 8192` for Moonlight. For MiniCPM, add `--template minicpm3-chat`
during data preparation. Reduce `--batch-size` or use `--prefill-chunk-tokens 2048`
if GPU memory is limited. Predictions and scores are saved in the output
directory, with a summary in `summary.json`.

## SGLang deployment

SGLang deployment supports DeepSeek-V2-Lite on one GPU, using packed INT2
content caches with BF16 positional keys and protected tokens.

```bash
conda create -n rolaq-serve python=3.12 -y
conda activate rolaq-serve
python -m pip install --upgrade pip
python deployment/install.py

python deployment/serve.py \
  --model models/deepseek --artifact runs/deepseek/artifact \
  --gpu 0 --port 30000 --context-length 16384
```

The installer downloads and patches the OSCAR/SGLang base in
`third_party/OSCAR` and installs the serving dependencies. Choose an idle GPU
and use the artifact calibrated for the same checkpoint.

Send a generation request:

```bash
curl http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":32}}'
```

Use `--method bf16` without `--artifact` to serve a BF16 reference. The server
runs in the foreground; stop it with Ctrl-C.

## Repository layout

```text
mla_quant/             calibration, artifact export, quantization, and evaluation
mla_quant/tasks/       HumanEval and MBPP evaluation adapters
scripts/prepare_ruler.py
                       RULER retrieval data preparation
deployment/install.py  pinned SGLang installation
deployment/serve.py    native INT2 server launcher
deployment/sglang/     RoLaQ parameter loader and fused INT2 writer
deployment/patches/    integration with the packed cache and attention readers
```

## License

Original code is licensed under Apache-2.0. Third-party notices and licenses for
OSCAR, SGLang, KVarN, RULER, and lm-evaluation-harness are retained in `NOTICE`
and `licenses/`.
