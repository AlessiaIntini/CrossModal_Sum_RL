#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=videoxum_dpo_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/dpo-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/dpo-%j.err

set -euo pipefail

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
VENV_DIR="/leonardo_scratch/fast/IscrC_HydroFM/aintini0/venvs/grpovideo"
CHECKPOINT_ROOT="/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/checkpoints-test"
RESULTS_ROOT="/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/results/dpo-test"
DATA_JSON="/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum/metadata/derived/train_videoxum_dpo.json"

cd "$PROJECT_DIR"
module load python/3.11.7
source "$VENV_DIR/bin/activate"

export HF_HOME="${SCRATCH}/GRPOVideo/cache/huggingface"
export TORCH_HOME="${SCRATCH}/GRPOVideo/cache/torch"
export PIP_CACHE_DIR="${SCRATCH}/GRPOVideo/cache/pip"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "$CHECKPOINT_ROOT" "$RESULTS_ROOT" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"

python -c "import av, scipy, sklearn, torch, transformers, peft, accelerate, qwen_vl_utils"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Python: $(which python)"
python --version
nvidia-smi

# Per eseguire solo la validation:
# sbatch --export=ALL,EVAL_CHECKPOINT=/percorso/checkpoint train_dpo_test.sh
MODE_ARGS=()
if [[ -n "${EVAL_CHECKPOINT:-}" ]]; then
    MODE_ARGS+=(--eval_only --eval_checkpoint "$EVAL_CHECKPOINT")
fi

srun python -u train_dpo.py \
    --data_json "$DATA_JSON" \
    --val_json /leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum/metadata/val_videoxum.json \
    --val_video_dir /leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum/val \
    --saveroot "$CHECKPOINT_ROOT" \
    --resultsroot "$RESULTS_ROOT" \
    --debug_samples 4000 \
    --eval_samples 2000 \
    --epochs 1 \
    --accum_steps 16 \
    --save_steps 20 \
    --lr 5e-6 \
    --beta 0.1 \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --lora_r 16 \
    --lora_dropout 0.05 \
    --max_pixels 100352 \
    --fps 1 \
    --max_frames 0 \
    --device cuda:0 \
    --resume_from /leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/checkpoints-test/Qwen2.5-VL-3B-Instruct/checkpoint-180 \
    "${MODE_ARGS[@]}"
