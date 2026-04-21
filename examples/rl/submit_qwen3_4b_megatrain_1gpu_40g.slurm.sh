#!/bin/bash

#SBATCH -o job.%j.out
#SBATCH -J grpo-qwen3-4b-40g
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# 禁用输出缓冲
export PYTHONUNBUFFERED=1

source ~/.bashrc

cd /home/u220320627/MegaTrain

# Activate your cluster runtime environment here.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-MegaTrain}"
conda activate "$CONDA_ENV_NAME"
# module load cuda/12.1

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
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "======================================"

mkdir -p logs

# 1) Prepare GSM8K parquet once if files do not exist.
if [[ ! -f "$HOME/data/gsm8k/train.parquet" || ! -f "$HOME/data/gsm8k/test.parquet" ]]; then
  python3 verl/examples/data_preprocess/gsm8k.py --local_save_dir "$HOME/data/gsm8k"
fi

# 2) Launch single-GPU MegaTrain RL for Qwen3-4B.
TRAIN_FILE="$HOME/data/gsm8k/train.parquet" \
TEST_FILE="$HOME/data/gsm8k/test.parquet" \
CUDA_VISIBLE_DEVICES=0 \
bash examples/rl/run_qwen3_4b_megatrain_1gpu_40g.sh
