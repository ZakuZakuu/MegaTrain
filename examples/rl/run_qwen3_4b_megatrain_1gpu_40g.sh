#!/bin/bash
# -----------------------------------------------------------------------------
# GRPO Training: Qwen3-4B on a Single A100 40GB via VERL + MegaTrain
# -----------------------------------------------------------------------------
# Goal: stable 1-GPU run with conservative memory settings for 40GB VRAM.
#
# Data preparation (run once, on login node or worker node):
#   python3 verl/examples/data_preprocess/gsm8k.py --local_save_dir $HOME/data/gsm8k
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen3_4b_megatrain_1gpu_40g.sh
#
# Optional overrides:
#   MODEL_PATH=/path/to/Qwen3-4B \
#   TRAIN_FILE=/path/to/train.parquet \
#   TEST_FILE=/path/to/test.parquet \
#   CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen3_4b_megatrain_1gpu_40g.sh
#
# Speed-up try (after stable run):
#   ... actor_rollout_ref.rollout.n=2 data.train_batch_size=4
# -----------------------------------------------------------------------------

set -x

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-4B"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"$HOME/data/gsm8k/test.parquet"}

PROJECT_NAME=${PROJECT_NAME:-"GRPO-Qwen3-4B-MegaTrain-40G"}
EXP_NAME=${EXP_NAME:-"grpo-4b-1gpu-40g"}
LOG_DIR=${LOG_DIR:-"logs"}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "${LOG_DIR}"

python3 -m verl.trainer.main_ppo \
    model_engine=megatrain \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=2 \
    data.max_prompt_length=384 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.trust_remote_code=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.strategy=megatrain \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.megatrain.checkpoint_interval=4 \
    actor_rollout_ref.actor.megatrain.num_grad_slabs=12 \
    actor_rollout_ref.actor.megatrain.max_seq_len=1024 \
    # flash_attention_2 requires the `flash-attn` package (FlashAttention 2, e.g. version 2.x)
    # and a compatible NVIDIA GPU architecture such as Ampere or newer.
    actor_rollout_ref.actor.megatrain.attn_implementation=flash_attention_2 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.strategy=megatrain \
    actor_rollout_ref.ref.use_torch_compile=False \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    # No default quantization for A100 40GB target; enable fp8 manually on compatible GPUs.
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    "$@" 2>&1 | tee "${LOG_DIR}/grpo-qwen3-4b-40g-${TIMESTAMP}.log"
