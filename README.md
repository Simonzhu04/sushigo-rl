# Sushi Go RL (3-Round Variant)

This repo implements a deterministic 2-player Sushi Go reinforcement-learning environment, baseline agents, PPO training and evaluation utilities, and an optional LLM-backed explain and coaching layer.

Current repo status:

- 3 rounds per game
- end-of-game pudding scoring
- chopsticks double-play actions
- random and heuristic opponents
- MaskablePPO training and evaluation utilities
- interactive terminal play against the RL model or heuristic
- optional LLM move explanations and coaching

The source-of-truth rules doc is [docs/RULES.md](docs/RULES.md).

## Game Summary

Implemented rules:

- 2 players
- 3 rounds
- `HAND_SIZE = 10`
- simultaneous drafting, then hand passing every turn
- fresh hands dealt each round from the fixed deck
- pudding accumulated across rounds and scored at game end
- chopsticks allows a two-card play and returns one chopsticks card to hand

Implemented card set:

- Tempura
- Sashimi
- Dumpling
- Maki 1 / 2 / 3
- Nigiri 1 / 2 / 3
- Wasabi
- Pudding
- Chopsticks

Scoring highlights:

- Tempura: `5` per pair
- Sashimi: `10` per triple
- Dumpling: capped progression `0,1,3,6,10,15`
- Wasabi affects the next later Nigiri in play order
- 2-player maki tie for first splits first place as `3` each
- pudding is scored once at game end with `penalty_for_last=False`

## Environment API

`SushiGoEnv` exposes:

- `reset(seed=None) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`

Observation:

- fixed-size `np.float32` vector
- current played counts for both players
- current hand counts plus fixed slot encoding
- turn, hand size, round index
- wasabi and maki summary features
- pudding counts
- accumulated round scores

Action space:

- `0..9`: play one card from the current hand index
- `10..99`: ordered chopsticks double-play actions

Useful helpers:

- `SushiGoEnv.decode_action_index(...)`
- `SushiGoEnv.cards_for_action(...)`
- `SushiGoEnv.describe_action(...)`

## Installation

From the repo root:

```bash
python -m pip install -e .
```

Or with Conda:

```bash
conda env create -f environment.yml
conda activate sushigo-rl
python -m pip install -e .
```

## Best Checkpoint

The current tracked best checkpoint in this repo is:

- `runs/repro_curriculum_1m.zip`
- `runs/repro_curriculum_1m.vecnormalize.pkl`

This model was freshly reproduced from the current training scripts with a 1,000,000-step curriculum run and matching `VecNormalize` statistics. Intermediate checkpoints at every 100k steps are in `runs/repro_curriculum_1m_checkpoints/`.

Confirmed deterministic evaluation on the current repo state, 1000 episodes:

- vs random: `96.6%` win rate, `+17.573` mean score diff
- vs heuristic: `95.7%` win rate, `+16.238` mean score diff

Baseline controls:

- heuristic vs random: `76.0%` win rate, `+7.911` mean score diff
- random vs random: near-balanced, `47.6%` win rate for player 1, `+0.065` mean score diff

For the exact training and evaluation commands, see [results/repro_curriculum_1m.md](results/repro_curriculum_1m.md).

## Ablation: Opponent Training Regime

Two ablation models (1M steps, seed 1) isolate the effect of curriculum training:

| Training regime | vs Random WR | vs Heuristic WR |
|----------------|-------------|----------------|
| Random-only | 0.968 | 0.866 |
| Heuristic-only | 0.955 | 0.955 |
| **Curriculum** | **0.966** | **0.957** |

Training against random only overfits to weak opponents (0.866 vs heuristic). Curriculum training achieves the strongest generalisation across both opponent types.

Ablation checkpoints:

- `runs/ablation_random_only.zip` + `runs/ablation_random_only.vecnormalize.pkl`
- `runs/ablation_heuristic_only.zip` + `runs/ablation_heuristic_only.vecnormalize.pkl`

## Training

Basic training:

```bash
python -m sushigo_rl.train --timesteps 200000 --seed 0 --model-out runs/latest
```

Common options:

- `--opponent-mode random|heuristic|mix|curriculum`
- `--mix-random-prob 0.5`
- `--curriculum-stages "0.0:200000,0.2:200000,0.5:200000"`
- `--vecnorm`
- `--vecnorm-path runs/latest.vecnormalize.pkl`
- `--checkpoint-every N`
- `--checkpoint-dir runs/checkpoints`
- `--tensorboard-log runs/tb`
- `--overfit-test`
- `--fixed-episode-seed`

Example long curriculum run matching the tracked best model:

```bash
python -m sushigo_rl.train \
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

## Evaluation

Evaluate the tracked best model:

```bash
python -m sushigo_rl.eval \
  --model runs/repro_curriculum_1m.zip \
  --vecnorm-path runs/repro_curriculum_1m.vecnormalize.pkl \
  --episodes 1000 \
  --opponents both \
  --baselines none \
  --deterministic
```

Run baseline-only matchups:

```bash
python -m sushigo_rl.eval --opponents none --baselines heuristic_vs_random --episodes 1000
python -m sushigo_rl.eval --opponents none --baselines random_vs_random --episodes 1000
```

Run the determinism sanity check:

```bash
python -m sushigo_rl.eval --opponents none --baselines none --repro-check
```

The evaluator reports:

- wins / losses / ties
- tie rate
- win rate including and excluding ties
- score-difference mean, std, and percentiles
- mean player and opponent score

## Play In The Terminal

Play against the RL checkpoint:

```bash
python -m sushigo_rl.cli_play --opponent rl
```

The CLI now defaults to `runs/repro_curriculum_1m.zip` and `runs/repro_curriculum_1m.vecnormalize.pkl`. The explicit form is:

```bash
python -m sushigo_rl.cli_play \
  --opponent rl \
  --model-path runs/repro_curriculum_1m.zip \
  --vecnorm-path runs/repro_curriculum_1m.vecnormalize.pkl
```

Play against the heuristic:

```bash
python -m sushigo_rl.cli_play --opponent heuristic
```

Interactive commands during play:

- enter an action index to play that move
- `help` to get coaching for your current turn
- `why` to explain the opponent's last move
- `quit` to exit

LLM-only autoplay mode:

```bash
python -m sushigo_rl.cli_play \
  --opponent rl \
  --llm-only
```

## LLM Setup

The coach and explain layer supports OpenAI, Gemini, or deterministic fallback templates.

By default:

- if an OpenAI key is present, OpenAI is preferred
- otherwise if a Gemini key is present, Gemini is used
- otherwise the repo falls back to built-in template text

Supported environment variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` with default `gpt-5.2`
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- `GEMINI_MODEL` with default `gemini-2.5-flash`
- `LLM_PROVIDER` with `auto|openai|gemini|fallback`

Example OpenAI setup:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-5.2
python -m sushigo_rl.cli_play --opponent rl
```

Example Gemini setup:

```bash
export GOOGLE_API_KEY=your_key_here
export GEMINI_MODEL=gemini-2.5-flash
python -m sushigo_rl.cli_play --opponent rl
```

Force fallback text even if API keys are set:

```bash
python -m sushigo_rl.cli_play --opponent rl --no-llm
```

## LLM Demo And Evaluation Commands

Generate move-by-move demo transcripts:

```bash
python -m sushigo_rl.llm_demo --episodes 2
```

Force demo fallback templates:

```bash
python -m sushigo_rl.llm_demo --episodes 2 --no-llm
```

Write the transcript to a specific file:

```bash
python -m sushigo_rl.llm_demo --episodes 1 --output-path runs/llm_logs/demo.txt
```

Quantitatively score explain and coach output:

```bash
python -m sushigo_rl.llm_eval --episodes 100 --mode both
```

Force a specific provider:

```bash
python -m sushigo_rl.llm_eval --provider openai
python -m sushigo_rl.llm_demo --provider gemini
```

## Tests

Run the full test suite:

```bash
pytest -q
```

Current status: `51 passed`.

The tests cover:

- scoring rules
- environment determinism and legality
- observation encoding
- LLM provider behavior
- LLM assistant helper behavior
