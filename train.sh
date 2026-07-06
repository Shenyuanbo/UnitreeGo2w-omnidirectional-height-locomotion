#!/bin/bash

NUM_ENVS=64
USE_TMUX=false
USE_WANDB=true

# 实验名和任务名设置
SESSION_NAME="go2w_height_train"
TASK_NAME="RobotLab-Isaac-Velocity-Height-Unitree-Go2W-v0"

while getopts "n:s" opt; do
  case $opt in
    n)
      NUM_ENVS=$OPTARG
      ;;
    s)
      USE_TMUX=true
      ;;
    *)
      echo "Usage: $0 [-n num_envs] [-s]"
      exit 1
      ;;
  esac
done

CMD="python scripts/reinforcement_learning/rsl_rl/train.py \
  --task ${TASK_NAME} \
  --num_envs ${NUM_ENVS} \
  --headless"

if [ "$USE_WANDB" = true ]; then
  CMD="${CMD} --logger wandb"
fi

echo "======================================"
echo "Task      : ${TASK_NAME}"
echo "Num envs  : ${NUM_ENVS}"
echo "Tmux      : ${USE_TMUX}"
echo "WandB     : ${USE_WANDB}"
echo "======================================"
echo "Command:"
echo "${CMD}"
echo "======================================"

if [ "$USE_TMUX" = true ]; then
  # 如果 session 已经存在，先提醒用户
  if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
    echo "Tmux session '${SESSION_NAME}' already exists."
    echo "Attach with:"
    echo "tmux attach -t ${SESSION_NAME}"
    exit 1
  fi

  tmux new-session -d -s ${SESSION_NAME} "${CMD}"

  echo "Started training in tmux session:"
  echo "  ${SESSION_NAME}"
  echo ""
  echo "Attach:"
  echo "  tmux attach -t ${SESSION_NAME}"
  echo ""
  echo "Detach without stopping:"
  echo "  Ctrl+B then D"
else
  eval ${CMD}
fi