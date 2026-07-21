# open-aht-observability

Stationary observability experiments for GPL (Graph-based Policy Learning) in Level-Based Foraging. The repository focuses on how sight radius and hidden teammate levels affect GPL and an auxiliary level-prediction objective.

## Setup

```bash
conda create -n observability-aht python=3.9 -y
conda activate observability-aht
pip install -r requirements.txt
```

Slurm scripts default to `MQP_CONDA_ENV=observability-aht`; override it if your cluster environment has a different name.

## What Is Kept

- `experiments/train_gpl.py`: stationary GPL baseline training.
- `experiments/train_gpl_inf.py`: stationary GPL training with an auxiliary teammate-level head.
- `experiments/eval_stationary.py`: greedy stationary checkpoint evaluation.
- `configs/gpl_lbf*.yaml`: full-observation, hidden-level, auxiliary-head, short-check, and sight-sweep configs.
- `scripts/slurm/`: Slurm entrypoints for baseline training, sight sweeps, short checks, and greedy evals.
- `scripts/plot_sight_sweep.py` and `scripts/plot_sight_sweep_figure.py`: observability figures.
- `tests/test_preprocess.py` and `tests/test_gpl_forward.py`: retained test coverage.

## Main Experiments

- `configs/gpl_lbf.yaml`: full-observation baseline (`sight=8`, teammate levels visible).
- `configs/gpl_lbf_q3_rw.yaml`: hidden-level baseline (`sight=3`, teammate levels hidden).
- `configs/gpl_lbf_q3_inf_aux.yaml`: hidden-level task with an auxiliary level-prediction head.
- `configs/gpl_lbf_q3_rw_sight{4,5,6,7}.yaml`: baseline sight sweep.
- `configs/gpl_lbf_q3_inf_aux_sight{4,5,6,7}.yaml`: auxiliary-head sight sweep.

## Run

```bash
# Train full-observation and sight=3 baselines
bash scripts/slurm/submit_training.sh

# Train sight sweeps and chain stationary greedy evals
bash scripts/slurm/submit_rw_sight_sweep.sh --with-eval
bash scripts/slurm/submit_sight_sweep.sh --with-eval

# Submit sight-sweep evals after training, if not chained
bash scripts/slurm/submit_rw_sight_sweep_eval.sh
bash scripts/slurm/submit_sight_sweep_eval.sh

# Plot completed sweeps
python scripts/plot_sight_sweep.py
python scripts/plot_sight_sweep_figure.py
```

Stationary greedy evaluation writes `stationary_eval_seed<seed>.csv` inside each run's `results/` directory.

## Test

```bash
python -m pytest tests/ -v
```
