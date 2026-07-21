#!/bin/bash
# Submit the Q3-inf-aux sight-sweep greedy evals (sight in {4,5,6,7}).
# Prerequisite: the matching training jobs have finished and written
# results/q3_inf_aux_sight{K}_rw_stationary/checkpoints/gpl_final.pt.
#
# Usage: bash scripts/slurm/submit_sight_sweep_eval.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

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

SIGHTS=(4 5 6 7)

for K in "${SIGHTS[@]}"; do
  CKPT="results/q3_inf_aux_sight${K}_rw_stationary/checkpoints/gpl_final.pt"
  if [[ ! -f "${CKPT}" ]]; then
    echo "WARNING: checkpoint missing for sight=${K}: ${CKPT} — skipping"
    continue
  fi
  echo "Submitting Q3-inf-aux greedy eval (sight=${K})..."
  sbatch --export=ALL "scripts/slurm/q3_inf_aux_sight${K}_greedy_eval.slurm"
done

echo "Done. squeue -u \$USER"
