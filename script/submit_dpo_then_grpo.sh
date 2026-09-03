#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/aintini/RL_VideoSum/GRPOVideo"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/Qwen2.5-VL-3B-Instruct}"
DPO_CHECKPOINT_ROOT="${DPO_CHECKPOINT_ROOT:-$PROJECT_DIR/outputs/dpo/checkpoints}"
MODEL_NAME="$(basename "$MODEL_DIR")"
DPO_FINAL="${DPO_FINAL:-$DPO_CHECKPOINT_ROOT/$MODEL_NAME/final}"

mkdir -p "$PROJECT_DIR/logs"
dpo_job_id="$(sbatch --parsable --export=ALL,MODEL_DIR="$MODEL_DIR",DPO_CHECKPOINT_ROOT="$DPO_CHECKPOINT_ROOT" "$PROJECT_DIR/train_dpo_hpc.sh")"
grpo_job_id="$(sbatch --parsable --dependency="afterok:$dpo_job_id" --export=ALL,MODEL_DIR="$MODEL_DIR",DPO_CHECKPOINT="$DPO_FINAL" "$PROJECT_DIR/train_grpo.sh")"

echo "DPO job:  $dpo_job_id"
echo "GRPO job: $grpo_job_id (partira' afterok:$dpo_job_id)"
echo "Checkpoint DPO atteso: $DPO_FINAL"
