# `repro_curriculum_1m` Reproducibility Note

This document records the training command, required artifacts, evaluation commands, and confirmed results for the fresh 1,000,000-step curriculum PPO checkpoint.

## Code revision

- Git commit used for training and evaluation: `63f777f0937a7c1124dceda48d41b0886447f953`
- Branch: `main`

## Required artifacts

- `runs/repro_curriculum_1m.zip`
- `runs/repro_curriculum_1m.vecnormalize.pkl`

These files must be used together.

## Environment

The commands below were run from the repository root using the existing Conda environment:

```bash
conda run -n sushigo-rl env PYTHONPATH=src python -m pytest -q
mkdir -p /tmp/matplotlib
```

`MPLCONFIGDIR=/tmp/matplotlib` was set for training and evaluation to avoid local matplotlib cache permission issues.

## Training command

```bash
PYTHONPATH=src /home/simon/miniconda3/envs/sushigo-rl/bin/python -m sushigo_rl.train \
  --opponent-mode curriculum \
  --curriculum-stages "0.0:200000,0.2:200000,0.5:300000,0.8:300000" \
  --timesteps 1000000 \
  --seed 0 \
  --vecnorm \
  --vecnorm-path runs/repro_curriculum_1m.vecnormalize.pkl \
  --model-out runs/repro_curriculum_1m \
  --checkpoint-every 100000 \
  --checkpoint-dir runs/repro_curriculum_1m_checkpoints \
  --checkpoint-prefix model \
  --verbose 0
```

## Test command

```bash
conda run -n sushigo-rl env PYTHONPATH=src python -m pytest -q
```

Observed result:

```text
51 passed
```

## Evaluation commands

Deterministic policy evaluation:

```bash
conda run -n sushigo-rl env PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib \
python -m sushigo_rl.eval \
  --model runs/repro_curriculum_1m.zip \
  --vecnorm-path runs/repro_curriculum_1m.vecnormalize.pkl \
  --episodes 1000 \
  --opponents both \
  --baselines none \
  --deterministic
```

Baseline controls:

```bash
conda run -n sushigo-rl env PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib \
python -m sushigo_rl.eval \
  --opponents none \
  --baselines heuristic_vs_random \
  --episodes 1000
```

```bash
conda run -n sushigo-rl env PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib \
python -m sushigo_rl.eval \
  --opponents none \
  --baselines random_vs_random \
  --episodes 1000
```

Reproducibility sanity check:

```bash
conda run -n sushigo-rl env PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib \
python -m sushigo_rl.eval \
  --opponents none \
  --baselines none \
  --repro-check \
  --repro-seed 17
```

## Confirmed results

Fresh 1M curriculum checkpoint, deterministic, 1000 episodes:

- Policy vs random: 966 wins, 23 losses, 11 ties
- Policy vs random win rate including ties: `0.966`
- Policy vs random mean score diff: `17.573`
- Policy vs heuristic: 957 wins, 35 losses, 8 ties
- Policy vs heuristic win rate including ties: `0.957`
- Policy vs heuristic mean score diff: `16.238`

Baseline controls, deterministic, 1000 episodes:

- Heuristic vs random: `0.760` win rate including ties, `7.911` mean score diff
- Random vs random: `0.476` win rate including ties, `0.065` mean score diff

Reproducibility sanity check:

- `repro_check: passed (seed=17, steps=30, final_my_score=42.0, final_opp_score=45.0, terminal_reward=-3.0)`

## Notes

- This checkpoint was trained from the current repository scripts rather than imported from an older workspace.
- The matching `VecNormalize` file is required for evaluation and CLI play.
- The checkpoint directory `runs/repro_curriculum_1m_checkpoints/` contains intermediate snapshots from the same run.
