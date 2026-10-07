#!/usr/bin/env bash
set -euo pipefail
# One model; four settings x three seeds. GPU selection uses CUDA_VISIBLE_DEVICES.
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 MODEL DATA_DIR ROTATION_DIR OUTPUT_DIR" >&2
  exit 2
fi
model="$1"
data_dir="$2"
rotation_dir="$3"
output_dir="$4"
for setting in full_joint kv_energy kv_weight cayley; do
  for seed in 42 43 44; do
    if [ -f "$output_dir/$setting/seed$seed/artifact/manifest.json" ]; then
      continue
    fi
    python -m mla_quant.train --model "$model" \
      --train-data "$data_dir/train.json" --validation-data "$data_dir/validation.json" \
      --rotation-dir "$rotation_dir" --setting "$setting" --gain-mode unit \
      --seed "$seed" --batch-order-seed "$seed" \
      --output "$output_dir/$setting/seed$seed"
  done
done
