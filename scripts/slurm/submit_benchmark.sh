#!/bin/bash
# Submit Observability Benchmark Sweeps across LBF and Wolfpack.
#
# Usage:
#   bash scripts/slurm/submit_benchmark.sh --dry-run             # Test run 1 job locally/smoke-test
#   bash scripts/slurm/submit_benchmark.sh --env lbf            # Launch 40 LBF jobs
#   bash scripts/slurm/submit_benchmark.sh --env wolfpack       # Launch 40 Wolfpack jobs
#   bash scripts/slurm/submit_benchmark.sh --env all            # Launch all 80 jobs

set -euo pipefail

ENV_TARGET="all"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_TARGET="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "=== Running Dry-Run Verification (Smoke Test) ==="
  echo "Testing LBF dry run..."
  BENCHMARK_ENV="lbf" SMOKE_TEST=1 SLURM_ARRAY_TASK_ID=0 SLURM_SUBMIT_DIR="${ROOT_DIR}" \
    bash scripts/slurm/benchmark_array.slurm
  echo "Testing Wolfpack dry run..."
  BENCHMARK_ENV="wolfpack" SMOKE_TEST=1 SLURM_ARRAY_TASK_ID=0 SLURM_SUBMIT_DIR="${ROOT_DIR}" \
    bash scripts/slurm/benchmark_array.slurm
  echo "Dry run completed successfully! Ready for cluster submission."
  exit 0
fi

submit_env() {
  local target_env="$1"
  echo "Submitting 40-job array for ${target_env} (4 sights x 2 teammates x 5 seeds)..."
  sbatch --export=ALL,BENCHMARK_ENV="${target_env}",SMOKE_TEST=0 \
         --array=0-39 \
         scripts/slurm/benchmark_array.slurm
}

if [ "${ENV_TARGET}" == "lbf" ]; then
  submit_env "lbf"
elif [ "${ENV_TARGET}" == "wolfpack" ]; then
  submit_env "wolfpack"
elif [ "${ENV_TARGET}" == "all" ]; then
  submit_env "lbf"
  submit_env "wolfpack"
else
  echo "Invalid env: ${ENV_TARGET}. Choose from 'lbf', 'wolfpack', 'all'."
  exit 1
fi

echo "Submission complete. Check status with: squeue -u \$USER"
