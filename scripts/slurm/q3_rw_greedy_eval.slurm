#!/bin/bash
# Q3_rw greedy stationary evaluation (500 episodes, epsilon=0 via stationary greedy evaluation)
# Info-nerf-only (3 agents, no force coop); same protocol as q1_greedy_eval.slurm.
# Results: results/q3_rw_stationary_greedy_eval_500/
#SBATCH --job-name=q3-rw-greedy-eval
#SBATCH --account=xliu14
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}" || exit 1
mkdir -p logs/slurm

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${MQP_CONDA_ENV:-observability-aht}"

export PYTHONUNBUFFERED=1

CHECKPOINT="results/q3_rw_stationary/checkpoints/gpl_final.pt"
RESULTS_DIR="results/q3_rw_stationary_greedy_eval_500"

echo "Q3_rw greedy stationary eval (500 episodes)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Results: ${RESULTS_DIR}"
nvidia-smi

python -u experiments/eval_stationary.py \
    --checkpoint "${CHECKPOINT}" \
    --config configs/gpl_lbf_q3_rw.yaml \
    --n-episodes 500 \
    --seed 42 \
    --results-dir "${RESULTS_DIR}"
