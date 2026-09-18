#!/bin/bash
#SBATCH --job-name=aht-distractor
#SBATCH --account=xliu14
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
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

MODE="${DISTRACTOR_MODE:-full}"  # "full" or "canary"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [ "${MODE}" == "canary" ]; then
  # 6 Canary runs: (0_none, 8_semantic, 8_null) x (lbf, wolfpack)
  COND_IDX=$(( TASK_ID % 3 ))
  ENV_IDX=$(( TASK_ID / 3 ))
  SEED=42
  N_EPISODES=100

  if [ "${ENV_IDX}" -eq 0 ]; then
    ENV_NAME="lbf"
    SIGHT=8
  else
    ENV_NAME="wolfpack"
    SIGHT=10
  fi

  if [ "${COND_IDX}" -eq 0 ]; then
    NUM_DIST=0
    DIST_TYPE="none"
  elif [ "${COND_IDX}" -eq 1 ]; then
    NUM_DIST=8
    DIST_TYPE="semantic"
  else
    NUM_DIST=8
    DIST_TYPE="null"
  fi
else
  # Full Matrix: 30 runs = 2 envs x 3 conditions x 5 seeds
  SEEDS=(42 43 44 45 46)
  SEED_IDX=$(( TASK_ID % 5 ))
  TEMP=$(( TASK_ID / 5 ))
  COND_IDX=$(( TEMP % 3 ))
  ENV_IDX=$(( TEMP / 3 ))

  SEED="${SEEDS[${SEED_IDX}]}"
  N_EPISODES=128000

  if [ "${ENV_IDX}" -eq 0 ]; then
    ENV_NAME="lbf"
    SIGHT=8
  else
    ENV_NAME="wolfpack"
    SIGHT=10
  fi

  if [ "${COND_IDX}" -eq 0 ]; then
    NUM_DIST=4
    DIST_TYPE="semantic"
  elif [ "${COND_IDX}" -eq 1 ]; then
    NUM_DIST=8
    DIST_TYPE="semantic"
  else
    NUM_DIST=8
    DIST_TYPE="null"
  fi
fi

TEAMMATE="random"
DEVICE="cuda"

OUTPUT_DIR="results/${ENV_NAME}_sight${SIGHT}_${TEAMMATE}_dist${NUM_DIST}_${DIST_TYPE}_seed${SEED}"
mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "Launching Distractor Task ${TASK_ID} on $(hostname)"
echo "Mode: ${MODE}"
echo "Environment: ${ENV_NAME} (Sight ${SIGHT})"
echo "Teammate: ${TEAMMATE}"
echo "Distractors: ${NUM_DIST} (${DIST_TYPE})"
echo "Seed: ${SEED}"
echo "Episodes: ${N_EPISODES}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=================================================="

python -u experiments/run_benchmark.py \
    --env "${ENV_NAME}" \
    --sight "${SIGHT}" \
    --teammate_type "${TEAMMATE}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --n-episodes "${N_EPISODES}" \
    --device "${DEVICE}" \
    --num_distractors "${NUM_DIST}" \
    --distractor_type "${DIST_TYPE}" \
    --distractor_seed "$(( SEED + 100000 ))"

echo "Distractor Task ${TASK_ID} completed successfully!"
