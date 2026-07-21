#!/bin/bash
# Submit the Q3-inf-aux sight sweep (sight in {4, 5, 6, 7}) in parallel.
# Launches 4 training jobs and, optionally, the matching greedy-eval jobs
# as chained dependencies.
#
# Usage:
#   bash scripts/slurm/submit_sight_sweep.sh            # trainings only
#   bash scripts/slurm/submit_sight_sweep.sh --with-eval   # + chained greedy evals
#
# Each training writes to results/q3_inf_aux_sight{K}_rw_stationary/.
# Each greedy eval (if requested) writes to
# results/q3_inf_aux_sight{K}_rw_stationary_greedy_eval_500/ and only
# starts after the matching training finishes (afterok dependency).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/slurm

WITH_EVAL=0
if [[ "${1:-}" == "--with-eval" ]]; then
  WITH_EVAL=1
fi

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
  echo "Submitting Q3-inf-aux training (sight=${K})..."
  TRAIN_OUT=$(sbatch --export=ALL "scripts/slurm/q3_inf_aux_sight${K}_train.slurm")
  echo "  ${TRAIN_OUT}"
  TRAIN_ID=$(echo "${TRAIN_OUT}" | awk '{print $NF}')

  if [[ "${WITH_EVAL}" -eq 1 ]]; then
    echo "Submitting chained greedy eval (sight=${K}, afterok:${TRAIN_ID})..."
    sbatch --export=ALL --dependency="afterok:${TRAIN_ID}" \
      "scripts/slurm/q3_inf_aux_sight${K}_greedy_eval.slurm"
  fi
done

echo
echo "Done. Monitor with: squeue -u \$USER"
if [[ "${WITH_EVAL}" -eq 0 ]]; then
  echo "After trainings finish, submit evals with:"
  echo "  bash scripts/slurm/submit_sight_sweep_eval.sh"
fi
