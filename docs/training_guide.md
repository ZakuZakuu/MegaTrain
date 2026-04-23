# MegaTrain 训练常用工具速查手册

## 1. tmux — 后台训练（断开 SSH 不中断）

```bash
# 创建新会话并启动训练
tmux new -s train
CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen2_5_1_5b_megatrain_1gpu_40g.sh

# 分离会话（训练继续后台运行）
# 按 Ctrl+B，然后按 D

# 重新连接
tmux attach -t train

# 查看所有会话
tmux ls

# 杀掉会话
tmux kill-session -t train
```

## 2. GPU 监控

```bash
# 单次查看
nvidia-smi

# 每 2 秒刷新
watch -n 2 nvidia-smi

# 查看 GPU 进程详情
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv

# gpustat（更美观，需安装）
pip install gpustat
gpustat -i 2
```

## 3. CPU / 内存监控

```bash
# 查看内存使用
free -h

# htop（交互式，需安装）
htop

# 查看 Python 进程内存
ps aux | grep python | grep -v grep

# 查看特定进程的内存详情
pmap -x <PID>
```

## 4. 日志查看

```bash
# 实时跟踪最新日志
tail -f logs/grpo-qwen2_5-1_5b-40g-*.log

# 只看训练指标（过滤进度条）
tail -f logs/grpo-qwen2_5-1_5b-40g-*.log | grep "step:"

# 只看验证结果
grep "val/" logs/grpo-qwen2_5-1_5b-40g-*.log | tail -20

# 搜索错误信息
grep -i "error\|exception\|traceback" logs/grpo-qwen2_5-1_5b-40g-*.log

# 查看最新日志最后 50 行
tail -50 logs/grpo-qwen2_5-1_5b-40g-$(ls -t logs/ | head -1 | sed 's/.*-//;s/.log//').log
```

## 5. TensorBoard 可视化

```bash
# 启动 TensorBoard（训练脚本需配置 trainer.logger=[console,tensorboard]）
pip install tensorboard
tensorboard --logdir=tensorboard_log --bind_all --port 6006

# 浏览器访问
# http://<服务器IP>:6006
```

## 6. Checkpoint 管理

```bash
# 查看已保存的 checkpoint
ls checkpoints/GRPO-Qwen2_5-1_5B-MegaTrain-40G/grpo-1_5b-1gpu-40g/

# 查看最新 checkpoint
cat checkpoints/GRPO-Qwen2_5-1_5B-MegaTrain-40G/grpo-1_5b-1gpu-40g/latest_checkpointed_iteration.txt

# 删除旧 checkpoint（参数变更后需要）
rm -rf checkpoints/GRPO-Qwen2_5-1_5B-MegaTrain-40G/grpo-1_5b-1gpu-40g/
```

## 7. 训练前后结果对比

```bash
# 对比不同 step 的验证指标
grep "val/acc\|val/reward" logs/grpo-qwen2_5-1_5b-40g-*.log

# 查看训练 throughput 变化
grep "perf/throughput" logs/grpo-qwen2_5-1_5b-40g-*.log

# 查看 reward 变化趋势
grep "critic/score/mean" logs/grpo-qwen2_5-1_5b-40g-*.log
```

## 8. 进程管理

```bash
# 查看所有 Python 进程
ps aux | grep python

# 杀掉训练进程（优雅停止）
kill <PID>

# 强制杀掉
kill -9 <PID>

# 查看 Ray 进程
ray list actors
```

## 9. Hydra 配置覆盖（不改脚本）

```bash
# 临时修改参数运行
CUDA_VISIBLE_DEVICES=0 bash examples/rl/run_qwen2_5_1_5b_megatrain_1gpu_40g.sh \
    data.train_batch_size=16 \
    actor_rollout_ref.rollout.n=2 \
    trainer.test_freq=5

# 查看完整配置
cat outputs/2026-04-23/02-40-40/.hydra/config.yaml
```

## 10. 常用排查命令

```bash
# 查看磁盘空间
df -h

# 查看 GPU 温度
nvidia-smi -q -d TEMPERATURE

# 查看 CUDA 版本
nvcc --version

# 查看 PyTorch CUDA 支持
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# 查看 GPU 计算能力
python3 -c "import torch; print(torch.cuda.get_device_capability(0))"
```
