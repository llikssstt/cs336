#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_train.sh                 # default lr list (1e-3)
#   bash run_train.sh 1e-3 5e-4 3e-4  # sweep lrs from args
#   LRS_LIST="1e-3 5e-4" bash run_train.sh
#
# Optional env overrides:
#   TRAIN_DATA, VOCAB_SIZE, CONTEXT_LENGTH, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF,
#   BATCH_SIZE, MAX_ITERS, WARMUP_ITERS, MIN_LR, WEIGHT_DECAY, MAX_GRAD_NORM,
#   LOG_INTERVAL, EVAL_INTERVAL, CHECKPOINT_INTERVAL, CHECKPOINT_ROOT,
#   WANDB_PROJECT, WANDB_RUN_PREFIX

if [[ $# -gt 0 ]]; then
    LRS=("$@")
elif [[ -n "${LRS_LIST:-}" ]]; then
    read -r -a LRS <<< "${LRS_LIST}"
else
    LRS=("5e-4")
fi

TRAIN_DATA=${TRAIN_DATA:-/mnt/d/cs336/data/TinyStoriesV2-GPT4-valid.npy}
VOCAB_SIZE=${VOCAB_SIZE:-10000}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-256}
D_MODEL=${D_MODEL:-512}
NUM_LAYERS=${NUM_LAYERS:-4}
NUM_HEADS=${NUM_HEADS:-16}
D_FF=${D_FF:-1344}

BATCH_SIZE=${BATCH_SIZE:-32}
MAX_ITERS=${MAX_ITERS:-5000}
WARMUP_ITERS=${WARMUP_ITERS:-500}
MIN_LR=${MIN_LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

LOG_INTERVAL=${LOG_INTERVAL:-50}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-5000}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-./checkpoints}

WANDB_PROJECT=${WANDB_PROJECT:-tinystories-lm}
WANDB_RUN_PREFIX=${WANDB_RUN_PREFIX:-tinystories}

common_args=(
    --train_data "$TRAIN_DATA"
    --vocab_size "$VOCAB_SIZE"
    --context_length "$CONTEXT_LENGTH"
    --d_model "$D_MODEL"
    --num_layers "$NUM_LAYERS"
    --num_heads "$NUM_HEADS"
    --d_ff "$D_FF"
    --batch_size "$BATCH_SIZE"
    --max_iters "$MAX_ITERS"
    --warmup_iters "$WARMUP_ITERS"
    --weight_decay "$WEIGHT_DECAY"
    --max_grad_norm "$MAX_GRAD_NORM"
    --log_interval "$LOG_INTERVAL"
    --eval_interval "$EVAL_INTERVAL"
    --checkpoint_interval "$CHECKPOINT_INTERVAL"
)

for lr in "${LRS[@]}"; do
    # If MIN_LR >= LR, automatically set min_lr = 0.1 * lr
    min_lr=$(uv run python - "${lr}" "${MIN_LR}" <<'PY'
import sys

lr = float(sys.argv[1])
min_lr = float(sys.argv[2])
if min_lr >= lr:
    min_lr = 0.1 * lr
print(min_lr)
PY
    )

    tag="lr_${lr}"
    tag="${tag//./p}"
    tag="${tag//-/_m}"
    tag="${tag//+/_p}"
    checkpoint_dir="${CHECKPOINT_ROOT}/${tag}"
    run_name="${WANDB_RUN_PREFIX}-${tag}-$(date +%Y%m%d_%H%M%S)"

    echo "==> Running with lr=${lr}, min_lr=${min_lr}"
    echo "    checkpoint_dir=${checkpoint_dir}"
    echo "    wandb_run_name=${run_name}"

    LR="${lr}" MIN_LR="${MIN_LR}" \
        uv run python -m cs336_basics.train \
            "${common_args[@]}" \
            --lr "${lr}" \
            --min_lr "${min_lr}" \
            --checkpoint_dir "${checkpoint_dir}" \
            --wandb_project "${WANDB_PROJECT}" \
            --wandb_run_name "${run_name}"
done