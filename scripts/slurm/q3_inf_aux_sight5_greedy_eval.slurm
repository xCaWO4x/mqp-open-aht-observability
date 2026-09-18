#!/bin/bash
# Q3-inf-aux SIGHT SWEEP greedy stationary eval — sight = 5 (500 eps, epsilon=0)
# Uses the trained checkpoint from q3_inf_aux_sight5_train.slurm.
# Results: results/q3_inf_aux_sight5_rw_stationary_greedy_eval_500/
#SBATCH --job-name=q3-inf-aux-s5-greedy
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

CHECKPOINT="results/q3_inf_aux_sight5_rw_stationary/checkpoints/gpl_final.pt"
RESULTS_DIR="results/q3_inf_aux_sight5_rw_stationary_greedy_eval_500"

echo "Q3-inf-aux sight=5 greedy stationary eval (500 episodes)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Results: ${RESULTS_DIR}"
nvidia-smi

python -u experiments/eval_stationary.py \
    --checkpoint "${CHECKPOINT}" \
    --config configs/gpl_lbf_q3_inf_aux_sight5.yaml \
    --n-episodes 500 \
    --seed 42 \
    --results-dir "${RESULTS_DIR}"
