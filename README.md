# Sushi Go (2-Player) Reinforcement Learning

A compact reinforcement-learning project implementing a **2-player Sushi Go drafting environment** (reduced card set) plus baselines and PPO training.

## Frozen artifacts

Stable policy artifacts are versioned under `artifacts/`:
- `artifacts/final_policy.zip`
- `artifacts/final_vecnorm.pkl`
- `artifacts/final_config.json`

Use them directly for eval/inference:

```bash
python -m sushigo_rl.eval \
  --model artifacts/final_policy.zip \
  --vecnorm-path artifacts/final_vecnorm.pkl \
  --episodes 500 \
  --opponents both \
  --deterministic
```

## Why this project
Sushi Go has delayed rewards (set collection), strategic drafting, and opponent interaction — a good testbed for RL beyond toy control tasks.

---

## Rules implemented (v1)

### Players / Round
- **2 players**
- **1 round**
- Each player is dealt `HAND_SIZE` cards.
- Each turn both players:
  1) Choose **one** card to play.
  2) Reveal simultaneously.
  3) Add chosen card to their own played pile.
  4) **Pass** the remaining hand to the other player (swap hands).
- Repeat until hands are empty.

### Card set included
- Tempura
- Sashimi
- Dumpling
- Maki: 1 / 2 / 3
- Nigiri: 1 / 2 / 3
- Wasabi

### Scoring

Let `my` be your played pile.

#### Tempura
- Score **5 points per pair**
- `score += 5 * floor(tempura_count / 2)`

#### Sashimi
- Score **10 points per triple**
- `score += 10 * floor(sashimi_count / 3)`

#### Dumpling
- Score increases with count (cap at 5):
  - 0→0, 1→1, 2→3, 3→6, 4→10, 5+→15
- Use: `dumpling_points = [0,1,3,6,10,15]` and `min(count,5)`

#### Nigiri + Wasabi
- Nigiri values: 1/2/3
- Wasabi: each Wasabi can be paired with the **next Nigiri you play after it** (in play order) to **triple** that Nigiri’s value.
- A Wasabi without a later Nigiri is worth 0.
- If you play Nigiri with no available earlier unpaired Wasabi, it scores its base value.

#### Maki
- Sum maki icons from cards: maki-1/2/3.
- Compare totals across players:
  - Player with **most** maki gets **6**.
  - Player with **second-most** gets **3**.
- **Ties split points evenly**:
  - Tie for first: split 6+3 = 9 between tied players (2 players => 4.5 each).
  - Tie for second (only possible with >2 players) is irrelevant here, but keep split logic generic if you want.
- Since we are 2-player:
  - If totals unequal: 6 to higher, 3 to lower.
  - If equal: each gets 4.5.

> Note: We keep scoring as floats internally due to 4.5. Report final scores as float or as half-points.

---

## Environment API

Gymnasium-like environment:

- `reset(seed: int | None) -> (obs, info)`
- `step(action: int) -> (obs, reward, terminated, truncated, info)`

### Action space
- Action is **index into current hand**: `0..hand_size-1`
- Invalid indices are masked (and should raise or be rejected).

### Observation
A fixed-size `np.float32` vector, plus an `action_mask` in `info` (or returned separately depending on wrapper).

Recommended observation contents (what we expect in v1):
1) **My played counts** for each card type (or feature group)
2) **Opponent visible played counts**
3) **Current hand counts** by card type
4) `turn_index`, `cards_left_in_hand`, maybe `maki_totals_so_far`

(Exact encoding is in code; keep it stable for training.)

### Reward
- Intermediate rewards: `0`
- Terminal reward: `my_total_score - opp_total_score`

This keeps learning aligned with winning.

---

## Deck / Dealing

We use a **fixed composition** deck (deterministic given seed).
For v1, pick a moderate deck that supports combos, e.g.:

- Tempura: 14
- Sashimi: 14
- Dumpling: 14
- Maki1: 6
- Maki2: 12
- Maki3: 8
- Nigiri1: 10
- Nigiri2: 5
- Nigiri3: 5
- Wasabi: 6

Total = 94 cards (we’ll sample without replacement to deal 2 * HAND_SIZE cards).

Default `HAND_SIZE = 10` (so 20 cards used per episode).

> Deck composition is fixed by `docs/RULES.md` for v1 reproducibility.

---

## Baseline agents

### Random
Picks uniformly among legal actions.

### Heuristic (suggested policy)
Deterministic scoring heuristic based on:
- Immediate points (Nigiri, completing Tempura pairs / Sashimi triples / Dumpling marginal gain)
- Wasabi synergy (prefer Nigiri3 if you have unpaired Wasabi already played; prefer Wasabi if you expect high Nigiri later)
- Maki race (if opponent has more maki so far, value maki higher)

This should be simple and fast, not “perfect play”.

---

## Training

Training uses `sb3-contrib` `MaskablePPO` with fixed-size `Discrete(HAND_SIZE)` actions and explicit action masking via `env.action_masks()`.

Default PPO hyperparameters:
- `gamma=0.99`
- `gae_lambda=0.95`
- `ent_coef=0.01`
- `learning_rate=3e-4`
- `n_steps=512`
- `batch_size=64`

### Install
From repo root:

```bash
python -m pip install -e .
```

### Opponent modes
`train.py` supports selecting one opponent policy per episode at reset:
- `--opponent-mode random`
- `--opponent-mode heuristic`
- `--opponent-mode mix --mix-random-prob 0.5`
- `--opponent-mode curriculum --curriculum-stages "0.0:200000,0.2:200000,0.5:200000"`

### Basic train/eval

```bash
pytest -q
python -m sushigo_rl.train --timesteps 200000 --seed 0 --model-out runs/latest
python -m sushigo_rl.eval --model runs/latest.zip --episodes 500 --seed 123 --opponents both
```

`eval.py` reports:
- wins/losses/ties
- tie rate
- win-rate including ties (`wins / episodes`)
- win-rate excluding ties (`wins / (wins + losses)`)
- score-diff mean/std and p05/p25/p50/p75/p95
- mean my score / opponent score

### Mask-debug smoke test

Enable strict mask assertions in the environment:

```bash
python -m sushigo_rl.train --timesteps 2000 --seed 0 --env-debug-mask --model-out runs/debug_mask_smoke
```

Equivalent env-var toggle:

```bash
SUSHIGO_ENV_DEBUG_MASK=1 python -m sushigo_rl.train --timesteps 2000 --seed 0 --model-out runs/debug_mask_smoke
```

### Random-only sanity experiment

Train against 100% random for 200k timesteps and evaluate:

```bash
python -m sushigo_rl.train --opponent-mode random --timesteps 200000 --seed 0 --model-out runs/random_only
python -m sushigo_rl.eval --model runs/random_only.zip --episodes 500 --seed 123 --opponents random
```

Convenience shortcut:

```bash
python -m sushigo_rl.train --random-only-experiment --seed 0
```

### VecNormalize (optional)

```bash
python -m sushigo_rl.train --timesteps 200000 --seed 0 --vecnorm --model-out runs/latest_vecnorm
python -m sushigo_rl.eval --model runs/latest_vecnorm.zip --episodes 500 --seed 123 --vecnorm-path runs/latest_vecnorm.vecnormalize.pkl
```

### Curriculum training + checkpoints + learning curve

```bash
# 1M-step curriculum training with VecNormalize and periodic checkpoints
python -m sushigo_rl.train \
  --opponent-mode curriculum \
  --curriculum-stages "0.0:200000,0.2:200000,0.5:300000,0.8:300000" \
  --timesteps 1000000 \
  --seed 0 \
  --model-out runs/curriculum_1m_slotobs_vecnorm \
  --vecnorm \
  --vecnorm-path runs/curriculum_1m_slotobs_vecnorm.pkl \
  --checkpoint-every 100000 \
  --checkpoint-dir runs/curriculum_1m_slotobs_vecnorm_checkpoints \
  --checkpoint-prefix model \
  --verbose 0

# Final policy eval (deterministic), 2000 episodes each
python -m sushigo_rl.eval \
  --model runs/curriculum_1m_slotobs_vecnorm.zip \
  --episodes 2000 \
  --seed 123 \
  --opponents both \
  --deterministic \
  --vecnorm-path runs/curriculum_1m_slotobs_vecnorm.pkl

# Checkpoint learning-curve sweep (500 episodes per checkpoint per opponent)
python -m sushigo_rl.eval_curve \
  --checkpoint-dir runs/curriculum_1m_slotobs_vecnorm_checkpoints \
  --checkpoint-prefix model \
  --episodes 500 \
  --seed 123 \
  --deterministic \
  --opponents both \
  --vecnorm-dir runs/curriculum_1m_slotobs_vecnorm_checkpoints \
  --csv-out runs/curriculum_1m_slotobs_vecnorm_checkpoints/learning_curve.csv \
  --plot-out runs/curriculum_1m_slotobs_vecnorm_checkpoints/learning_curve.png
```

### Eval diagnostics and baselines

```bash
# Reproducibility sanity check: identical trajectories for same seed + actions
python -m sushigo_rl.eval --opponents none --repro-check --repro-seed 17 --repro-actions 0,0,0,0,0,0,0,0,0,0

# Random vs random baseline
python -m sushigo_rl.eval --episodes 1000 --opponents none --baselines random_vs_random --seed 123

# Heuristic vs random baseline
python -m sushigo_rl.eval --episodes 1000 --opponents none --baselines heuristic_vs_random --seed 123

# Policy vs itself baseline
python -m sushigo_rl.eval --model runs/latest.zip --episodes 1000 --opponents none --baselines policy_vs_self --seed 123
```

### Fixed-seed overfit sanity mode

```bash
python -m sushigo_rl.train --overfit-test --seed 0
```

This runs:
- random-only training for 50k timesteps
- fixed episode seed each reset (no deck variation)
- post-train eval on the same seed for 200 episodes

## LLM assistant (Explain + Coach)

The interactive CLI can call an external LLM for:
- `help` (coach mode): top-3 move suggestions with tradeoffs
- `why` (explain mode): explanation of the opponent agent's previous move

Provider selection supports `openai`, `gemini`, or `fallback` (`auto` by default).
If no provider key is available, it uses deterministic fallback templates.

### Configure API key

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-5.2"   # optional override

# Gemini Developer API (Google AI Studio)
export GOOGLE_API_KEY="your_gemini_key_here"   # or GEMINI_API_KEY
export GEMINI_MODEL="gemini-2.0-flash"         # optional override
```

### Run CLI

```bash
# Human vs frozen RL opponent
python -m sushigo_rl.cli_play

# Human vs heuristic opponent
python -m sushigo_rl.cli_play --opponent heuristic

# Force offline fallback mode
python -m sushigo_rl.cli_play --no-llm

# Non-interactive LLM-only mode (auto-plays top recommendation)
python -m sushigo_rl.cli_play --llm-only --no-llm
```

Assistant responses are logged to JSONL in `runs/llm_logs/`.

### Example transcript snippet

```text
LLM mode: fallback templates (no OPENAI_API_KEY or no SDK client).
Assistant logs: runs/llm_logs/assistant_20260217_203000.jsonl

Turn 1/10
Your hand:
  [0] wasabi
  [1] maki3
  [2] tempura
Choose move [index/help/why/quit]: help
Top moves (fallback coach):
1) idx 1 -> maki3 (p=0.612): affects maki race (current 0 vs 0).
2) idx 0 -> wasabi (p=0.228): setup card; strongest if you can follow with high nigiri soon.
3) idx 2 -> tempura (p=0.160): better when you can complete/keep a tempura pair.
```

## LLM Explanation Quality

Use `llm_eval.py` to compute lightweight quality metrics over generated explanations/coaching:
- average explanation length (`avg_words`)
- strategic term mention rate (`key_term_rate`) for terms like `pair`, `maki`, `wasabi`
- diversity ratio (`distinct_ratio`)

Results are appended to:
- `runs/llm_logs/llm_evaluation.csv`

### Run quantitative evaluation

```bash
# Fallback-only evaluation (no external API)
python -m sushigo_rl.llm_eval --episodes 100 --mode both --no-llm

# Auto provider selection (prefers OpenAI key, then Gemini key)
python -m sushigo_rl.llm_eval --episodes 100 --mode both

# Force Gemini
python -m sushigo_rl.llm_eval --episodes 100 --mode both --provider gemini
```

### Run 5-episode LLM demo transcript

```bash
# API-backed demo (requires OPENAI_API_KEY)
python -m sushigo_rl.llm_demo --episodes 5

# Gemini-backed demo
python -m sushigo_rl.llm_demo --episodes 5 --provider gemini

# Offline fallback demo
python -m sushigo_rl.llm_demo --episodes 5 --no-llm
```

Demo transcripts are written to:
- `runs/llm_logs/llm_demo_<timestamp>.txt`

### Qualitative strategy patterns to look for

- Mentions of `maki` race when deciding between maki cards and point cards.
- Mentions of `wasabi` setup vs immediate nigiri value.
- Mentions of set-completion pressure (`tempura` pairs, `sashimi` triples, `dumpling` progression).
- Use of action probabilities to justify confidence between top alternatives.

Free-tier note:
- Gemini Flash free-tier usage can hit temporary rate limits.
- The provider layer uses retries with exponential backoff + jitter, plus in-run caching keyed by state/prompt/model to reduce duplicate calls in `llm_eval`.

### Comparison of template fallback vs LLM API explanations

Latest local run summary (`runs/llm_logs/llm_evaluation.csv`):

- `explain`:
  - fallback baseline: `avg_words=31.000`, `key_term_rate=1.000`, `distinct_ratio=0.951`
  - API-attempt run: `avg_words=31.006`, `key_term_rate=1.000`, `distinct_ratio=0.952`
- `coach`:
  - fallback baseline: `avg_words=42.942`, `key_term_rate=1.000`, `distinct_ratio=0.950`
  - API-attempt run: `avg_words=42.942`, `key_term_rate=1.000`, `distinct_ratio=0.950`

Note:
- In this environment, API calls returned connectivity/quota errors, so `fallback_mode=True` in the API-attempt rows.
- When API access is healthy, expect `fallback_mode=False` and richer variation in wording.
