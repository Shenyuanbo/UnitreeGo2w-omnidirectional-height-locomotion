#!/bin/bash
set -euo pipefail

TASK=${TASK:-RobotLab-Isaac-Velocity-Height-Unitree-Go2W-v0}
DEFAULT_ROOT=${DEFAULT_ROOT:-logs/rsl_rl/unitree_go2w_height}

if [[ $# -gt 0 && "$1" != -* ]]; then
  LOG_DIR=$1
  shift
else
  LOG_DIR=$(ls -td ${DEFAULT_ROOT}/* 2>/dev/null | head -n 1 || true)
fi

RECORD=0
VIDEO_LENGTH=2000
NUM_ENVS=1
CHECKPOINT=""
FOLLOW_CAMERA=1
DROP_HEIGHT_COMMAND=${DROP_HEIGHT_COMMAND:-0}

usage() {
  echo "Usage: $0 [log_dir] [-v] [-l video_length] [-n num_envs] [-c checkpoint] [-- extra_hydra_overrides...]"
}

while getopts ":vl:n:c:h" opt; do
  case $opt in
    v) RECORD=1 ;;
    l) VIDEO_LENGTH=$OPTARG ;;
    n) NUM_ENVS=$OPTARG ;;
    c) CHECKPOINT=$OPTARG ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

if [[ -z "${LOG_DIR}" || ! -d "${LOG_DIR}" ]]; then
  echo "[ERROR] Log directory not found: ${LOG_DIR}"
  exit 1
fi

if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT=$(ls -v "${LOG_DIR}"/model_*.pt 2>/dev/null | tail -n 1 || true)
fi

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

COMMON_ARGS=(
  --task "${TASK}"
  --checkpoint "${CHECKPOINT}"
  --num_envs "${NUM_ENVS}"
  --headless
)

if [[ ${FOLLOW_CAMERA} -eq 1 ]]; then
  COMMON_ARGS+=(--follow-camera)
fi

if [[ ${DROP_HEIGHT_COMMAND} -eq 1 ]]; then
  COMMON_ARGS+=(env.observations.policy.height_commands=null)
fi

echo "=============================="
echo "TASK         : ${TASK}"
echo "LOG_DIR      : ${LOG_DIR}"
echo "CHECKPOINT   : ${CHECKPOINT}"
echo "NUM_ENVS     : ${NUM_ENVS}"
echo "RECORD       : ${RECORD}"
echo "VIDEO_LENGTH : ${VIDEO_LENGTH}"
echo "FOLLOW_CAMERA: ${FOLLOW_CAMERA}"
echo "DROP_HEIGHT_COMMAND: ${DROP_HEIGHT_COMMAND}"
echo "=============================="

if [[ ${RECORD} -eq 1 ]]; then
  rm -rf "${LOG_DIR}/videos/play"
  python scripts/reinforcement_learning/rsl_rl/play.py \
    "${COMMON_ARGS[@]}" \
    --video \
    --video_length "${VIDEO_LENGTH}" \
    "$@"
else
  python scripts/reinforcement_learning/rsl_rl/play.py \
    "${COMMON_ARGS[@]}" \
    "$@"
fi
