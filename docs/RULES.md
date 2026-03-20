# Sushi Go RL Rules Specification

This document defines the implemented rules for the `sushigo-rl-guanfang` environment.

It is a deterministic 2-player, 3-round Sushi Go variant intended for reinforcement learning experiments.

## 1. Game configuration

- Players: exactly 2
- Learning agent: player 0
- Opponent: player 1
- Rounds: 3
- Default hand size: `10`
- All randomness must come from the environment RNG and respect `reset(seed=...)`

## 2. Deck composition

The fixed deck contains:

- Tempura: 14
- Sashimi: 14
- Dumpling: 14
- Maki-1: 6
- Maki-2: 12
- Maki-3: 8
- Nigiri-1: 10
- Nigiri-2: 5
- Nigiri-3: 5
- Wasabi: 6
- Pudding: 10
- Chopsticks: 4

Total cards: `108`

At the start of each round:

- shuffle the full fixed deck with the environment RNG
- deal `HAND_SIZE` cards to each player
- unused cards remain out of play for that round

## 3. Turn structure

Each round lasts exactly `HAND_SIZE` turns.

Per turn:

1. both players choose an action simultaneously
2. actions resolve simultaneously
3. played cards are added to the current round pile in play order
4. remaining hands are swapped
5. turn counter increments

After `HAND_SIZE` turns:

- the round is scored
- current-round played piles reset
- a fresh round starts unless this was round 3

## 4. Actions

The environment uses a fixed discrete action space.

For `HAND_SIZE = 10`:

- actions `0..9` mean: play the card at that hand index
- the remaining actions encode ordered two-card chopsticks plays

Chopsticks action requirements:

- the player must already have a `chopsticks` card in their current round played pile
- the player must have at least 2 cards in hand
- the chosen two indices must be distinct
- after the two-card play resolves, one `chopsticks` card is removed from the table and returned to hand

The environment must expose an action mask for legal actions.

## 5. Scoring

### 5.1 Tempura

- `5` points per pair

### 5.2 Sashimi

- `10` points per triple

### 5.3 Dumpling

- cumulative progression: `0, 1, 3, 6, 10, 15`
- capped at 5 or more dumplings

### 5.4 Nigiri and Wasabi

- Nigiri values are `1`, `2`, and `3`
- Wasabi triples the next later Nigiri in play order
- Wasabi does not affect Nigiri played before it

### 5.5 Maki

- Maki icons are summed within the round
- higher total gets `6`
- lower total gets `3`
- if tied for most in 2-player mode, each player gets `3`

### 5.6 Pudding

- Pudding is not scored per round
- all pudding cards are accumulated across the full 3-round game
- at game end, the player with more pudding gets `+6`
- this implementation uses `penalty_for_last=False`, so the player with fewer pudding does not receive `-6`
- tied pudding counts score `0`

## 6. Rewards

- intermediate reward: `0`
- terminal reward: `final_my_score - final_opp_score`

`final_*_score` includes:

- all round scores
- end-of-game pudding scoring

## 7. Observations

Observations are fixed-size `np.float32` vectors containing:

- my current-round played counts
- opponent current-round played counts
- my current hand counts
- hand slot one-hot encoding
- current turn
- current hand size
- my unpaired wasabi count
- my maki icon count
- opponent maki icon count
- round index
- my pudding count
- opponent pudding count
- my accumulated round score
- opponent accumulated round score

## 8. Determinism requirements

The following must hold:

- same seed produces the same initial state
- same seed plus same action sequence produces the same trajectory
- final reward equals `my_score - opp_score`
- terminal scores agree with rules-module scoring helpers
