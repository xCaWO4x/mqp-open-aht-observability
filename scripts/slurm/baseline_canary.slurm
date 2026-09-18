#!/bin/bash
#SBATCH --job-name=canary-baselines
#SBATCH --account=xliu14
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}" || exit 1
mkdir -p logs/slurm

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if conda env list | grep -q "myenv"; then
  conda activate "myenv"
elif conda env list | grep -q "observability-aht"; then
  conda activate "observability-aht"
else
  conda activate base
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# 4 Canary Configurations:
# Task 0: Transformer-Q on LBF (sight 3, random, seed 1)
# Task 1: IPPO-GRU on LBF (sight 3, random, seed 1)
# Task 2: Transformer-Q on Wolfpack (sight 4, random, seed 1)
# Task 3: IPPO-GRU on Wolfpack (sight 4, random, seed 1)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [ "${TASK_ID}" -eq 0 ]; then
  ALGO="transformer_q"
  ENV_NAME="lbf"
  SIGHT="3"
  TEAMMATE="random"
  SEED="1"
elif [ "${TASK_ID}" -eq 1 ]; then
  ALGO="ippo_gru"
  ENV_NAME="lbf"
  SIGHT="3"
  TEAMMATE="random"
  SEED="1"
elif [ "${TASK_ID}" -eq 2 ]; then
  ALGO="transformer_q"
  ENV_NAME="wolfpack"
  SIGHT="4"
  TEAMMATE="random"
  SEED="1"
elif [ "${TASK_ID}" -eq 3 ]; then
  ALGO="ippo_gru"
  ENV_NAME="wolfpack"
  SIGHT="4"
  TEAMMATE="random"
  SEED="1"
else
  echo "Invalid Task ID: ${TASK_ID}"
  exit 1
fi

OUTPUT_DIR="results/${ENV_NAME}_${ALGO}_sight${SIGHT}_${TEAMMATE}_seed${SEED}"
mkdir -p "${OUTPUT_DIR}"

DEVICE="${BENCHMARK_DEVICE:-cpu}"

echo "=================================================="
echo "Launching Canary Task ${TASK_ID} on $(hostname)"
echo "Algorithm:   ${ALGO}"
echo "Environment: ${ENV_NAME}"
echo "Sight:       ${SIGHT}"
echo "Teammate:    ${TEAMMATE}"
echo "Seed:        ${SEED}"
echo "Device:      ${DEVICE}"
echo "Output Dir:  ${OUTPUT_DIR}"
echo "=================================================="

python -u train.py \
    --algo "${ALGO}" \
    --env "${ENV_NAME}" \
    --sight_radius "${SIGHT}" \
    --teammate_type "${TEAMMATE}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --train_steps 2000000 \
    --eval_interval 25000 \
    --device "${DEVICE}"

echo "Canary Task ${TASK_ID} completed successfully!"
