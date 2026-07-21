#!/bin/bash
# Q3_rw SIGHT SWEEP — sight = 4 training run
# Info-nerf-only baseline (no aux head). Uses train_gpl.py.
# Config:  configs/gpl_lbf_q3_rw_sight4.yaml
# Results: results/q3_rw_sight4_stationary/
#SBATCH --job-name=q3-rw-s4
#SBATCH --account=xliu14
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX6000B
#SBATCH --time=1-00:00:00
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

echo "Q3_rw sight sweep: sight=4 (no aux)"
nvidia-smi

python -u experiments/train_gpl.py \
    --config configs/gpl_lbf_q3_rw_sight4.yaml
