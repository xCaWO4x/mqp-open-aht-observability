#!/bin/bash
#SBATCH --job-name=marl-baselines
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

ALGO="${ALGO:-transformer_q}"
BENCHMARK_ENV="${BENCHMARK_ENV:-lbf}"

# Sights array based on environment
if [ "${BENCHMARK_ENV}" == "lbf" ]; then
  SIGHTS=(3 4 6 8)
else
  SIGHTS=(3 4 6 10)
fi

TEAMMATES=("random" "greedy")
SEEDS=(1 2 3 4 5)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

N_SEEDS=${#SEEDS[@]}
N_TEAMMATES=${#TEAMMATES[@]}
N_SIGHTS=${#SIGHTS[@]}

SEED_IDX=$(( TASK_ID % N_SEEDS ))
TEMP=$(( TASK_ID / N_SEEDS ))
TM_IDX=$(( TEMP % N_TEAMMATES ))
SIGHT_IDX=$(( TEMP / N_TEAMMATES ))

if [ "${SIGHT_IDX}" -ge "${N_SIGHTS}" ]; then
  echo "Task ID ${TASK_ID} out of range for ${BENCHMARK_ENV}"
  exit 0
fi

SIGHT="${SIGHTS[${SIGHT_IDX}]}"
TEAMMATE="${TEAMMATES[${TM_IDX}]}"
SEED="${SEEDS[${SEED_IDX}]}"

OUTPUT_DIR="results/${BENCHMARK_ENV}_${ALGO}_sight${SIGHT}_${TEAMMATE}_seed${SEED}"
mkdir -p "${OUTPUT_DIR}"

DEVICE="${BENCHMARK_DEVICE:-cpu}"

echo "=================================================="
echo "Launching Baseline Task ${TASK_ID} on $(hostname)"
echo "Algorithm:   ${ALGO}"
echo "Environment: ${BENCHMARK_ENV}"
echo "Sight:       ${SIGHT}"
echo "Teammate:    ${TEAMMATE}"
echo "Seed:        ${SEED}"
echo "Device:      ${DEVICE}"
echo "Output Dir:  ${OUTPUT_DIR}"
echo "=================================================="

python -u train.py \
    --algo "${ALGO}" \
    --env "${BENCHMARK_ENV}" \
    --sight_radius "${SIGHT}" \
    --teammate_type "${TEAMMATE}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --train_steps 2000000 \
    --eval_interval 50000 \
    --device "${DEVICE}"

echo "Task ${TASK_ID} completed successfully!"
