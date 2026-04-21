#!/bin/bash
#SBATCH --job-name=grpo-qwen3-4b-40g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -euo pipefail

# Optional, set based on your cluster policy:
# #SBATCH --partition=<your_partition>
# #SBATCH --account=<your_account>

cd MegaTrain

# Activate your cluster runtime environment here.
# source /path/to/venv/bin/activate
# module load cuda/12.1

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
