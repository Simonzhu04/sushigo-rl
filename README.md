# Sushi Go RL (3-Round Variant)

This repo implements a deterministic 2-player Sushi Go reinforcement-learning environment, baseline agents, PPO training/evaluation utilities, and an LLM-backed explain/coach layer.

Current repo status:

- 3 rounds per game
- end-of-game pudding scoring
- chopsticks double-play actions
- random and heuristic opponents
- MaskablePPO training and evaluation utilities
- interactive CLI play against the RL model or heuristic
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

The tracked best checkpoint in this repo is:

- `runs/guanfang_best_20260316.zip`
- `runs/guanfang_best_20260316.vecnormalize.pkl`

Fresh confirmation results on the current repo state:

- deterministic eval, 1000 episodes vs random: `94.7%` win rate, `+15.320` mean score diff
- deterministic eval, 1000 episodes vs heuristic: `85.4%` win rate, `+11.473` mean score diff
- stochastic eval, 1000 episodes vs random: `92.2%` win rate, `+14.184` mean score diff
- stochastic eval, 1000 episodes vs heuristic: `80.6%` win rate, `+9.387` mean score diff

For the exact commands and recorded results, see [results/guanfang_best_20260316.md](results/guanfang_best_20260316.md).

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

Example curriculum run:

```bash
python -m sushigo_rl.train \
  --opponent-mode curriculum \
  --curriculum-stages "0.0:100000,0.3:100000,0.6:100000" \
  --timesteps 300000 \
  --vecnorm \
  --vecnorm-path runs/curriculum_300k_decay.vecnormalize.pkl \
  --model-out runs/curriculum_300k_decay
```

## Evaluation

Evaluate the tracked best model:

```bash
python -m sushigo_rl.eval \
  --model runs/guanfang_best_20260316.zip \
  --vecnorm-path runs/guanfang_best_20260316.vecnormalize.pkl \
  --episodes 500 \
  --opponents both \
  --deterministic
```

Run baseline-only matchups:

```bash
python -m sushigo_rl.eval --opponents none --baselines heuristic_vs_random --episodes 500
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
python -m sushigo_rl.cli_play \
  --opponent rl \
  --model-path runs/guanfang_best_20260316.zip \
  --vecnorm-path runs/guanfang_best_20260316.vecnormalize.pkl
```

Play against the heuristic:

```bash
python -m sushigo_rl.cli_play \
  --opponent heuristic \
  --model-path runs/guanfang_best_20260316.zip \
  --vecnorm-path runs/guanfang_best_20260316.vecnormalize.pkl
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
  --model-path runs/guanfang_best_20260316.zip \
  --vecnorm-path runs/guanfang_best_20260316.vecnormalize.pkl \
  --llm-only
```

## LLM Setup

The coach/explain layer supports OpenAI, Gemini, or deterministic fallback templates.

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
python -m sushigo_rl.cli_play \
  --opponent rl \
  --model-path runs/guanfang_best_20260316.zip \
  --vecnorm-path runs/guanfang_best_20260316.vecnormalize.pkl
```

Example Gemini setup:

```bash
export GOOGLE_API_KEY=your_key_here
export GEMINI_MODEL=gemini-2.5-flash
python -m sushigo_rl.cli_play \
  --opponent rl \
  --model-path runs/guanfang_best_20260316.zip \
  --vecnorm-path runs/guanfang_best_20260316.vecnormalize.pkl
```

Force fallback text even if API keys are set:

```bash
python -m sushigo_rl.cli_play \
  --opponent rl \
  --model-path runs/guanfang_best_20260316.zip \
  --vecnorm-path runs/guanfang_best_20260316.vecnormalize.pkl \
  --no-llm
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

Quantitatively score explain/coach output:

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
