# Agent Instructions (Codex)

## Authoritative Rule Source

The implementation MUST strictly follow:

    docs/RULES.md

`docs/RULES.md` is the single source of truth for:
- Game mechanics
- Deck composition
- Scoring logic
- Reward definition
- Termination conditions

If any ambiguity exists, follow RULES.md exactly.
Do NOT invent, modify, or extend rules beyond what is defined there.

---

## Goal

Implement a 2-player Sushi Go reinforcement learning project in Python with:

- A deterministic Gymnasium-style environment
- Correct scoring for the reduced card set defined in RULES.md
- Baseline agents (random + heuristic)
- PPO training (stable-baselines3)
- Evaluation script (win-rate + score)
- Unit tests for rules + environment legality

---

## Scope (must-follow)

1) **2-player only**
2) **1 round only**
3) Implement EXACTLY the cards defined in `docs/RULES.md`
4) No additional mechanics (no pudding, no chopsticks, no variants)
5) Environment must be deterministic with seeding
6) Observation must be:
   - Fixed-size vector (np.float32)
   - Separate action mask (np.bool_)
7) Actions must be index into the current hand
8) API must provide:

   reset(seed=...)  
   step(action)  

   returning:
   (obs, reward, terminated, truncated, info)

---

## Implementation Order (important)

Codex should implement in the following order:

1) rules.py
   - Card definitions
   - Scoring logic
   - Pure deterministic functions only
   - Must match RULES.md exactly

2) tests/test_rules.py
   - Unit tests verifying scoring edge cases from RULES.md examples

3) env.py
   - Deterministic reset
   - Deterministic deck shuffle using seed
   - Simultaneous action resolution
   - Hand swapping
   - Proper termination after HAND_SIZE turns
   - Action masking
   - Reward ONLY at terminal state

4) tests/test_env.py
   - Deterministic reset
   - Illegal action rejection
   - Correct episode length
   - Final score consistency with rules.py

5) Baseline agents

6) PPO training + evaluation

Do NOT skip directly to PPO before tests pass.

---

## Repo structure (preferred)

- src/sushigo_rl/rules.py
- src/sushigo_rl/env.py
- src/sushigo_rl/agents/random_agent.py
- src/sushigo_rl/agents/heuristic_agent.py
- src/sushigo_rl/train.py
- src/sushigo_rl/eval.py
- tests/test_rules.py
- tests/test_env.py

---

## Environment Details (must implement)

Reset:
- Build deck using exact composition in RULES.md
- Shuffle deterministically using provided seed
- Deal HAND_SIZE cards per player

Step:
- Validate action using action mask
- Opponent action selected via provided policy callable
- Remove chosen cards
- Add to played piles
- Swap remaining hands
- Increment turn counter

Episode ends after exactly HAND_SIZE turns.

Reward:
- Intermediate reward = 0
- Terminal reward = my_score − opp_score
- Must match RULES.md scoring exactly

Info dict MUST include:
- my_score
- opp_score
- turn
- hand_size
- last_actions
- optionally card breakdowns

---

## Scoring (Critical)

All scoring MUST:

- Follow docs/RULES.md exactly
- Be deterministic
- Be test-covered
- Support fractional maki tie values (4.5)

Scoring logic must be implemented independently of environment mechanics.

---

## Baseline Agents

Random agent:
- Uniform over legal actions

Heuristic agent:
- Deterministic rule-based selection
- Use only visible state information
- No randomness unless seeded
- Must not modify environment

---

## RL Training

- Prefer stable-baselines3 PPO
- Use sb3-contrib MaskablePPO if available
- If not, implement action masking safely
- Train against mixture of:
  - 50% random
  - 50% heuristic
- Configurable opponent mix

Do not modify game rules for reward shaping.

---

## Testing (Mandatory)

Unit tests must verify:

Rules:
- Tempura pairs
- Sashimi triples
- Dumpling progression
- Wasabi application order
- Maki tie split logic

Environment:
- Same seed → identical initial hands
- Invalid action rejected
- Episode length equals HAND_SIZE
- Final score equals rules.score_total()

All tests must pass before PPO implementation.

---

## Code Quality

- Type hints everywhere
- Use dataclasses for state representation
- Keep functions small and readable
- Add docstrings
- Avoid unnecessary complexity
- Follow clean modular design

---

## Rule Freeze

RULES.md defines Version 1 of the game.

No rule modifications are allowed during RL implementation.

If extending later (e.g., adding pudding or multiple rounds), that must be done in a separate version.
