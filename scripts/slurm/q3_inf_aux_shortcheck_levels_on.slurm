#!/bin/bash
# Sanity check: can the aux head learn when the privileged level signal is
# actually in the observation? Identical to q3_inf_aux_shortcheck.slurm but
# with observe_agent_levels=true (agent feat dim 3 -> obs_dim=12).
# Pass: raw CE should collapse toward 0 within a few hundred episodes.
# Fail: raw CE stays near ln(3) -> would indicate a code-side bug.
# Config:  configs/gpl_lbf_q3_inf_aux_shortcheck_levels_on.yaml
# Results: results/q3_inf_aux_shortcheck_levels_on/
#SBATCH --job-name=q3-inf-aux-sc-levels-on
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

echo "Q3-inf-aux SANITY CHECK: observe_agent_levels=true, aux head should learn"
nvidia-smi

python -u experiments/train_gpl_inf.py \
    --config configs/gpl_lbf_q3_inf_aux_shortcheck_levels_on.yaml
