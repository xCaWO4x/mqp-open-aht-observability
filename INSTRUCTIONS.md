# Handoff: open-aht-observability

This repository is a focused copy for stationary observability experiments in LBF. It keeps the training, evaluation, plotting, and Slurm files needed for the sight-radius and auxiliary-level-head study.

## Core Question

How do teammate observability, sight radius, and hidden/visible levels affect GPL performance and learned type inference?

## Experiment Map

- `Q1`: full-observation GPL baseline, `configs/gpl_lbf.yaml`.
- `Q3_rw`: hidden-level GPL baseline, `configs/gpl_lbf_q3_rw.yaml`.
- `Q3-inf-aux`: hidden-level GPL with auxiliary level-prediction loss, `configs/gpl_lbf_q3_inf_aux.yaml`.
- Sight sweeps: `sight in {4,5,6,7}` for both `Q3_rw` and `Q3-inf-aux`.
- Short checks: auxiliary weight sweep plus levels-visible sanity check.

## Useful Commands

```bash
bash scripts/slurm/submit_training.sh
bash scripts/slurm/submit_greedy_eval.sh
bash scripts/slurm/submit_rw_sight_sweep.sh --with-eval
bash scripts/slurm/submit_sight_sweep.sh --with-eval
python scripts/plot_sight_sweep.py
python scripts/plot_sight_sweep_figure.py
python -m pytest tests/ -v
```

## Notes

- Slurm scripts use account `xliu14` and default to `MQP_CONDA_ENV=observability-aht`.
- Generated outputs belong under `results/`, `runs/`, `logs/`, and `figures/`.
- The auxiliary-head forward pass must reuse the carried LSTM hidden state; do not change it back to `hidden=None`.
