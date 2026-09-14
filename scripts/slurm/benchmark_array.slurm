#!/bin/bash
#SBATCH --job-name=aht-benchmark
#SBATCH --account=xliu14
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
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

if [ -n "${MQP_CONDA_ENV:-}" ]; then
  conda activate "${MQP_CONDA_ENV}"
elif conda env list | grep -q "observability-aht"; then
  conda activate "observability-aht"
elif conda env list | grep -q "myenv"; then
  conda activate "myenv"
else
  conda activate base
fi

export PYTHONUNBUFFERED=1

BENCHMARK_ENV="${BENCHMARK_ENV:-lbf}"
SMOKE_TEST="${SMOKE_TEST:-0}"

# Sights array based on environment
if [ "${BENCHMARK_ENV}" == "lbf" ]; then
  SIGHTS=(3 4 6 8)
else
  SIGHTS=(3 4 6 10)
fi

TEAMMATES=("random" "greedy")
SEEDS=(42 43 44 45 46)

# Compute parameter index from SLURM_ARRAY_TASK_ID (0-indexed or 1-indexed)
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

OUTPUT_DIR="results/${BENCHMARK_ENV}_sight${SIGHT}_${TEAMMATE}_seed${SEED}"
mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "Launching Task ${TASK_ID} on $(hostname)"
echo "Environment: ${BENCHMARK_ENV}"
echo "Sight: ${SIGHT}"
echo "Teammate: ${TEAMMATE}"
echo "Seed: ${SEED}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=================================================="

EXTRA_ARGS=""
if [ "${SMOKE_TEST}" == "1" ]; then
  EXTRA_ARGS="--smoke-test"
fi

DEVICE="${BENCHMARK_DEVICE:-cuda}"

python -u experiments/run_benchmark.py \
    --env "${BENCHMARK_ENV}" \
    --sight "${SIGHT}" \
    --teammate_type "${TEAMMATE}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --n-episodes 128000 \
    --device "${DEVICE}" \
    ${EXTRA_ARGS}

echo "Task ${TASK_ID} completed successfully!"
