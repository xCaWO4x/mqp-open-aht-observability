#!/bin/bash
# Q3-inf-aux aux_weight=2.0 short check (12k eps). Same protocol as
# q3_inf_aux_shortcheck.slurm; only aux_weight / seed / output dirs differ.
# Config:  configs/gpl_lbf_q3_inf_aux_shortcheck_w2p0.yaml
# Results: results/q3_inf_aux_shortcheck_w2p0/
#SBATCH --job-name=q3-inf-aux-sc-w2p0
#SBATCH --account=xliu14
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00:00
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

echo "Q3-inf-aux SHORT CHECK, aux_weight=2.0 (12k eps)"
nvidia-smi

python -u experiments/train_gpl_inf.py \
    --config configs/gpl_lbf_q3_inf_aux_shortcheck_w2p0.yaml
