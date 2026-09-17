#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=grpo_test_chain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/infer-grpo-chain-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/infer-grpo-chain-%j.err

# Inferenza GRPO sul test a blocchi nella QOS di debug (30 min, max 2 job per utente),
# stessa logica di infer_dpo_test_chain_dbg.sh. Circa 12 s/video: 110 video per
# blocco lasciano margine al caricamento del modello. Gli altri parametri
# (CHECKPOINT, MAX_NEW_TOKENS, SUMMARY_STYLE, SUMMARY_ONLY, ...) passano per
# ambiente a infer_grpo_test_cineca.sh e vengono propagati ai blocchi successivi.
#
# Avvio: CHAIN_START=1670 MAX_NEW_TOKENS=256 SUMMARY_STYLE=concise sbatch script/infer_grpo_test_chain_dbg.sh

set -Eeuo pipefail

CHAIN_START="${CHAIN_START:?CHAIN_START richiesto}"
CHAIN_STOP="${CHAIN_STOP:-4000}"
CHUNK_SIZE="${CHUNK_SIZE:-110}"
NEXT_START=$((CHAIN_START + CHUNK_SIZE))
SCRIPT_PATH="/leonardo/home/userexternal/aintini0/GRPOVideo/script/infer_grpo_test_chain_dbg.sh"

echo "[chain] job=${SLURM_JOB_ID:-local} blocco=[$CHAIN_START, $((NEXT_START < CHAIN_STOP ? NEXT_START : CHAIN_STOP)))"
if (( NEXT_START < CHAIN_STOP )); then
    next_job=$(CHAIN_START=$NEXT_START CHAIN_STOP=$CHAIN_STOP CHUNK_SIZE=$CHUNK_SIZE \
        sbatch --parsable --export=ALL --dependency="afterany:${SLURM_JOB_ID}" "$SCRIPT_PATH")
    echo "[chain] blocco successivo da $NEXT_START sottomesso: job $next_job"
else
    echo "[chain] ultimo blocco"
fi

TEST_START_INDEX="$CHAIN_START" \
TEST_SAMPLES=$(( (NEXT_START < CHAIN_STOP ? NEXT_START : CHAIN_STOP) - CHAIN_START )) \
    bash script/infer_grpo_test_cineca.sh
