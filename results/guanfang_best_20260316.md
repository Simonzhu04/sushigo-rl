# `guanfang_best_20260316` Reproducibility Note

This document records the checkpoint files, code revision, evaluation commands, and confirmed results for the `guanfang_best_20260316` policy.

## Code revision

- Git commit: `9bfda724eb7f9edc980acf3ba6258f168b356349`
- Branch: `main`

## Required artifacts

- `runs/guanfang_best_20260316.zip`
- `runs/guanfang_best_20260316.vecnormalize.pkl`

These files must be used together.

## Environment

The commands below were run from the repository root using the existing Conda environment:

```bash
conda run -n sushigo-rl env PYTHONPATH=src python -m pytest -q
mkdir -p /tmp/matplotlib
```

`MPLCONFIGDIR=/tmp/matplotlib` was set for evaluation to avoid local matplotlib cache permission issues.

## Test command

```bash
conda run -n sushigo-rl env PYTHONPATH=src python -m pytest -q
```

Observed result:

```text
51 passed in 4.64s
```

## Evaluation commands

Deterministic policy evaluation:

```bash
conda run -n sushigo-rl env PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib \
python -m sushigo_rl.eval \
  --model runs/guanfang_best_20260316.zip \
  --vecnorm-path runs/guanfang_best_20260316.vecnormalize.pkl \
  --episodes 1000 \
  --opponents both \
  --baselines none \
  --deterministic
```

Stochastic policy evaluation:

```bash
conda run -n sushigo-rl env PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib \
python -m sushigo_rl.eval \
  --model runs/guanfang_best_20260316.zip \
  --vecnorm-path runs/guanfang_best_20260316.vecnormalize.pkl \
  --episodes 1000 \
  --opponents both \
  --baselines none
```

## Confirmed results

Deterministic:

- Policy vs random: 947 wins, 42 losses, 11 ties
- Policy vs random win rate including ties: `0.947`
- Policy vs random mean score diff: `15.320`
- Policy vs heuristic: 854 wins, 120 losses, 26 ties
- Policy vs heuristic win rate including ties: `0.854`
- Policy vs heuristic mean score diff: `11.473`

Stochastic:

- Policy vs random: 922 wins, 64 losses, 14 ties
- Policy vs random win rate including ties: `0.922`
- Policy vs random mean score diff: `14.184`
- Policy vs heuristic: 806 wins, 167 losses, 27 ties
- Policy vs heuristic win rate including ties: `0.806`
- Policy vs heuristic mean score diff: `9.387`

## Notes

- This checkpoint loads cleanly with the current environment and matching VecNormalize file.
- These are fresh rerun metrics captured on March 20, 2026, not a historical March 16, 2026 evaluation log.
