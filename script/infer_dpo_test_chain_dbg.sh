#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=dpo_test_chain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/infer-dpo-chain-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/infer-dpo-chain-%j.err

# Inferenza DPO sul test a blocchi nella QOS di debug (30 min, max 2 job per utente).
# Ogni blocco, appena parte, sottomette il successivo con afterany: in coda ci
# sono al piu' 2 job e la catena prosegue anche se un blocco va in timeout.
# Gli indici non completati (timeout/blocco) si recuperano a fine catena
# confrontando i predictions-*.jsonl.
#
# Avvio: CHAIN_START=2836 sbatch script/infer_dpo_test_chain_dbg.sh

set -Eeuo pipefail

CHAIN_START="${CHAIN_START:?CHAIN_START richiesto}"
CHAIN_STOP="${CHAIN_STOP:-4000}"
CHUNK_SIZE="${CHUNK_SIZE:-120}"
NEXT_START=$((CHAIN_START + CHUNK_SIZE))
SCRIPT_PATH="/leonardo/home/userexternal/aintini0/GRPOVideo/script/infer_dpo_test_chain_dbg.sh"

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
    bash script/infer_dpo_test_cineca.sh
