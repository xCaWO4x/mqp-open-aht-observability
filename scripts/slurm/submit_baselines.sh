#!/bin/bash
# Submit MARL Baseline Sweeps (Transformer-Q and IPPO-GRU).
#
# Usage:
#   bash scripts/slurm/submit_baselines.sh --canary             # Launch 4 canary runs (1 per algo/env combination)
#   bash scripts/slurm/submit_baselines.sh --full               # Launch all full 40-run arrays (160 runs total)
#   bash scripts/slurm/submit_baselines.sh --algo transformer_q --env lbf # Launch specific combination (40 runs)

set -euo pipefail

MODE="canary"
ALGO_ARG="all"
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
    --algo)
      ALGO_ARG="$2"
      shift 2
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
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs/slurm

if [ "${MODE}" == "canary" ]; then
  echo "Submitting 4-canary job array (Task 0..3)..."
  sbatch --array=0-3 scripts/slurm/baseline_canary.slurm
  echo "Canaries submitted! Check with 'squeue -u \$(whoami)'"
  exit 0
fi

submit_combination() {
  local algo="$1"
  local env="$2"
  echo "Submitting 40-run array for Algo: ${algo} | Env: ${env}..."
  sbatch --export=ALL,ALGO="${algo}",BENCHMARK_ENV="${env}" \
         --array=0-39 \
         scripts/slurm/baseline_array.slurm
}

ALGOS=()
if [ "${ALGO_ARG}" == "all" ]; then
  ALGOS=("transformer_q" "ippo_gru")
else
  ALGOS=("${ALGO_ARG}")
fi

ENVS=()
if [ "${ENV_ARG}" == "all" ]; then
  ENVS=("lbf" "wolfpack")
else
  ENVS=("${ENV_ARG}")
fi

for a in "${ALGOS[@]}"; do
  for e in "${ENVS[@]}"; do
    submit_combination "${a}" "${e}"
  done
done

echo "All baseline sweeps submitted! Check with 'squeue -u \$(whoami)'"
