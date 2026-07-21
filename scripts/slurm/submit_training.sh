#!/bin/bash
# Submit Q1 and Q3_rw training jobs.
# Usage: bash scripts/slurm/submit_training.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

# Conda env on compute nodes: override with MQP_CONDA_ENV=myenv (etc.).
# If unset, pick the first env directory that exists among common names.
if [ -z "${MQP_CONDA_ENV:-}" ]; then
  for cand in observability-aht myenv; do
    for prefix in "${HOME}/miniconda3/envs" "${HOME}/anaconda3/envs"; do
      if [ -d "${prefix}/${cand}" ]; then
        MQP_CONDA_ENV="$cand"
        break 2
      fi
    done
  done
fi
: "${MQP_CONDA_ENV:=observability-aht}"
export MQP_CONDA_ENV
echo "Using conda env: ${MQP_CONDA_ENV}"

echo "Submitting Q1 (baseline training)..."
sbatch --export=ALL scripts/slurm/q1_train.slurm

echo "Submitting Q3_rw training..."
sbatch --export=ALL scripts/slurm/q3_rw_train.slurm

echo "Done. squeue -u \$USER"
