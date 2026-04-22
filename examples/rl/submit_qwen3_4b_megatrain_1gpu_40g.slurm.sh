#!/bin/bash

#SBATCH -o job.%j.out
#SBATCH -J grpo-qwen3-4b-40g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# 禁用输出缓冲
export PYTHONUNBUFFERED=1

source ~/.bashrc

cd $HOME/MegaTrain

# Activate your cluster runtime environment here.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-MegaTrain}"
conda activate "$CONDA_ENV_NAME"
# module load cuda/12.1

# Triton/DeepSpeed JIT needs a modern GCC toolchain (C11 headers like stdatomic.h).
if command -v module >/dev/null 2>&1; then
	module load gcc-10.1.0-gcc-4.8.5-2new4ox 2>/dev/null || module load gcc-10.1.0 2>/dev/null || true
fi
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"

# Fail fast if GCC is too old, otherwise job will die later in Triton JIT.
GCC_MAJOR="$($CC -dumpversion 2>/dev/null | cut -d. -f1)"
if [[ -z "$GCC_MAJOR" || "$GCC_MAJOR" -lt 9 ]]; then
	echo "[FATAL] GCC >= 9 required for Triton/DeepSpeed JIT, current: $CC ($($CC --version | head -n1 2>/dev/null))"
	echo "[FATAL] Please load a newer GCC module, e.g. gcc-10.1.0"
	exit 1
fi

# VERL expects CUDA/HIP device env vars to be mutually exclusive.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Make local verl package importable in this job environment.
export PYTHONPATH="$PWD/verl:${PYTHONPATH:-}"
python -m pip install -e ./verl --no-deps >/dev/null 2>&1 || true

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "=========================================="

# 打印环境信息
echo "========== Environment Info =========="
nvidia-smi
which python
python --version
conda info --envs
pip show torch
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-not set}"
echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-not set}"
echo "CC: ${CC}"
echo "CXX: ${CXX}"
$CC --version | head -n1
echo "======================================"
# Should be:
# CUDA_VISIBLE_DEVICES: 0
# ROCR_VISIBLE_DEVICES: not set
# HIP_VISIBLE_DEVICES: not set

mkdir -p logs

export HF_ENDPOINT=https://hf-mirror.com

# 1) Prepare GSM8K parquet once if files do not exist.
DATA_DIR="$HOME/data/gsm8k"
TRAIN_PARQUET="$DATA_DIR/train.parquet"
TEST_PARQUET="$DATA_DIR/test.parquet"

needs_rebuild=0
if [[ ! -f "$TRAIN_PARQUET" || ! -f "$TEST_PARQUET" ]]; then
	needs_rebuild=1
else
	# Verify schema for RLHFDataset: each parquet must include "prompt".
	if ! python3 - <<'PY'
import os
import sys
import datasets

files = [
		os.path.expanduser("~/data/gsm8k/train.parquet"),
		os.path.expanduser("~/data/gsm8k/test.parquet"),
]

ok = True
for f in files:
		ds = datasets.load_dataset("parquet", data_files=f, split="train[:1]")
		cols = set(ds.column_names)
		if "prompt" not in cols:
				print(f"[DatasetCheck] Missing 'prompt' in {f}. Columns: {sorted(cols)}")
				ok = False

sys.exit(0 if ok else 1)
PY
	then
		needs_rebuild=1
	fi
fi

if [[ "$needs_rebuild" -eq 1 ]]; then
	echo "Preprocessing dataset (build or schema-fix)..."
	rm -f "$TRAIN_PARQUET" "$TEST_PARQUET"
	python3 verl/examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR"
fi
echo "Data ready: $TRAIN_PARQUET and $TEST_PARQUET"

# 2) Launch single-GPU MegaTrain RL for Qwen3-4B.
TRAIN_FILE="$TRAIN_PARQUET" \
TEST_FILE="$TEST_PARQUET" \
bash examples/rl/run_qwen3_4b_megatrain_1gpu_40g.sh