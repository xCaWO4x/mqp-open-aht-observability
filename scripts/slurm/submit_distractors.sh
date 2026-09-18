#!/bin/bash
# Submit Observation Distractor Experiments for LBF and Wolfpack.
#
# Usage:
#   bash scripts/slurm/submit_distractors.sh --canary    # Launch 6 canary runs (0, 8-semantic, 8-null for LBF and Wolfpack)
#   bash scripts/slurm/submit_distractors.sh --full      # Launch all 30 full-length 128k-episode runs (3 conditions x 5 seeds x 2 envs)
#   bash scripts/slurm/submit_distractors.sh --env lbf   # Launch 15 runs for LBF only
#   bash scripts/slurm/submit_distractors.sh --env wolfpack # Launch 15 runs for Wolfpack only

set -euo pipefail

MODE="canary"
ENV_ARG="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --canary)
      MODE="canary"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --env)
      ENV_ARG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/distractor_array.slurm"
chmod +x "${SLURM_SCRIPT}"

if [ "${MODE}" == "canary" ]; then
  echo "Submitting 6 Distractor Canary runs (100 episodes each)..."
  export DISTRACTOR_MODE="canary"
  sbatch --array=0-5 "${SLURM_SCRIPT}"
elif [ "${MODE}" == "full" ]; then
  if [ "${ENV_ARG}" == "all" ]; then
    echo "Submitting all 30 Distractor runs (Tasks 0-29)..."
    export DISTRACTOR_MODE="full"
    sbatch --array=0-29 "${SLURM_SCRIPT}"
  elif [ "${ENV_ARG}" == "lbf" ]; then
    echo "Submitting 15 LBF Distractor runs (Tasks 0-14)..."
    export DISTRACTOR_MODE="full"
    sbatch --array=0-14 "${SLURM_SCRIPT}"
  elif [ "${ENV_ARG}" == "wolfpack" ]; then
    echo "Submitting 15 Wolfpack Distractor runs (Tasks 15-29)..."
    export DISTRACTOR_MODE="full"
    sbatch --array=15-29 "${SLURM_SCRIPT}"
  fi
fi

echo "Submission complete."
