# Sushi Go RL (3-Round Variant)

This project implements a 2-player Sushi Go reinforcement-learning environment with:

- 3 rounds per game
- end-of-game pudding scoring
- chopsticks double-play actions
- random and heuristic opponents
- MaskablePPO training and evaluation utilities
- optional LLM explain/coach tooling

It is a fuller game variant than the reduced 1-round prototype.

## Game rules implemented

- 2 players
- 3 rounds
- `HAND_SIZE = 10`
- each turn both players draft simultaneously, then pass hands
- fresh hands are dealt each round
- pudding is accumulated across rounds and scored at game end
- chopsticks allows a player to play two cards in one turn, then return the chopsticks card to hand

Implemented card set:

- Tempura
- Sashimi
- Dumpling
- Maki 1 / 2 / 3
- Nigiri 1 / 2 / 3
- Wasabi
- Pudding
- Chopsticks

Scoring notes:

- Tempura: `5` per pair
- Sashimi: `10` per triple
- Dumpling: standard capped progression `0,1,3,6,10,15`
- Wasabi applies to the next later Nigiri in play order
- 2-player maki ties use the official-style split for first only: `3` each
- pudding is scored only once at game end
- this project uses `penalty_for_last=False` for pudding, so the player with fewer puddings does not lose points

The source-of-truth rules doc is [docs/RULES.md](docs/RULES.md).

## Environment API

`SushiGoEnv` exposes a Gymnasium-style API:

- `reset(seed=None) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`

Observation contents include:

- my played counts for the current round
- opponent played counts for the current round
- current hand counts
- fixed hand-slot encoding
- turn index
- current hand size
- wasabi and maki summary features
- round index
- pudding counts
- accumulated round scores before final pudding scoring

## Action space

The environment uses a fixed discrete action space of size `100` when `HAND_SIZE = 10`.

- `0..9`: play one card from the corresponding hand index
- `10..99`: ordered chopsticks actions representing two card indices

Helper methods:

- `SushiGoEnv.decode_action_index(...)`
- `SushiGoEnv.cards_for_action(...)`
- `SushiGoEnv.describe_action(...)`

## Installation

From the repo root:

```bash
python -m pip install -e .
```

Or create the conda environment:

```bash
conda env create -f environment.yml
conda activate sushigo-rl
```

## Training

Basic training:

```bash
python -m sushigo_rl.train --timesteps 200000 --seed 0 --model-out runs/latest
```

Useful options:

- `--opponent-mode random|heuristic|mix|curriculum`
- `--vecnorm`
- `--checkpoint-every N`
- `--tensorboard-log runs/tb`
- `--overfit-test`

Example curriculum run:

```bash
python -m sushigo_rl.train \
  --opponent-mode curriculum \
  --curriculum-stages "0.0:100000,0.3:100000,0.6:100000" \
  --timesteps 300000 \
  --vecnorm \
  --model-out runs/curriculum_300k_decay
```

## Evaluation

Evaluate a model against random and heuristic opponents:

```bash
python -m sushigo_rl.eval \
  --model runs/curriculum_300k_decay.zip \
  --episodes 500 \
  --opponents both
```

If the model was trained with `VecNormalize`, also pass `--vecnorm-path`.

The evaluator reports:

- wins / losses / ties
- tie rate
- win rate including and excluding ties
- score-difference mean, std, and percentiles
- mean player and opponent score

## CLI play and LLM tools

Play in the terminal:

```bash
python -m sushigo_rl.cli_play --opponent heuristic
```

Generate LLM demo transcripts:

```bash
python -m sushigo_rl.llm_demo --no-llm --episodes 2
```

## Tests

Run:

```bash
pytest -q
```

The current tests cover rules, environment behavior, and LLM helper behavior.
