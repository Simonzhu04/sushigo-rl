# Sushi Go – 2 Player Reduced Version (RL Project Specification)

This document defines the exact rules that MUST be implemented.
The environment implementation MUST strictly follow this specification.

This is NOT the full commercial rulebook.
This is a reduced, deterministic RL-focused version.

---

# 1. Game Configuration

## Players
- Exactly 2 players.
- Player 0 = learning agent.
- Player 1 = opponent.

## Rounds
- 1 round only.
- No pudding.

## Determinism
- All randomness MUST use the environment seed.
- Reset with the same seed MUST produce identical initial hands and deck order.

---

# 2. Deck Composition

The deck MUST contain exactly the following cards:

Tempura: 14  
Sashimi: 14  
Dumpling: 14  
Maki-1: 6  
Maki-2: 12  
Maki-3: 8  
Nigiri-1 (Egg): 10  
Nigiri-2 (Salmon): 5  
Nigiri-3 (Squid): 5  
Wasabi: 6  

Total cards = 94

At the start of each episode:
- Shuffle full deck using provided RNG seed.
- Deal HAND_SIZE cards to each player.
- Remaining cards are unused.

Default:
HAND_SIZE = 10

Thus:
- 20 cards used per episode
- 74 unused

---

# 3. Turn Structure

Each episode consists of exactly HAND_SIZE turns.

At each turn:

1. Both players choose one card from their current hand.
2. Choices are revealed simultaneously.
3. Each chosen card is added to that player's played pile.
4. The remaining cards in each hand are swapped between players.
5. Continue to next turn.

After HAND_SIZE turns:
- Both hands are empty.
- Round ends.
- Scores are computed.

---

# 4. Action Rules

- Action is an index into the current hand.
- Only indices < current hand size are legal.
- Selecting an invalid action MUST raise an error or be rejected.
- Environment MUST provide an action mask.

---

# 5. Scoring Rules

Scoring occurs ONLY at the end of the round.

All scoring must be deterministic and follow exactly the rules below.

---

## 5.1 Tempura

- 5 points per pair.
- Extra unpaired cards give 0.

Formula:
floor(tempura_count / 2) * 5

Examples:
- 0 → 0
- 1 → 0
- 2 → 5
- 3 → 5
- 4 → 10

---

## 5.2 Sashimi

- 10 points per triple.
- Extra unpaired cards give 0.

Formula:
floor(sashimi_count / 3) * 10

Examples:
- 0 → 0
- 2 → 0
- 3 → 10
- 6 → 20

---

## 5.3 Dumpling

Dumpling scoring is cumulative:

Count → Points
0 → 0
1 → 1
2 → 3
3 → 6
4 → 10
5 or more → 15

If count > 5, score = 15.

---

## 5.4 Nigiri

Nigiri values:
Nigiri-1 → 1
Nigiri-2 → 2
Nigiri-3 → 3

---

## 5.5 Wasabi

Wasabi modifies Nigiri scoring.

Rule:

- When a Wasabi card is played, it becomes "available".
- The NEXT Nigiri card played by that player AFTER that Wasabi is worth triple its value.
- Each Wasabi can only affect one Nigiri.
- If multiple Wasabi are available, they apply in order played.
- Wasabi does NOT retroactively affect Nigiri played before it.
- Wasabi without a later Nigiri gives 0 points.

Implementation requirement:

Nigiri scoring MUST consider play order.

Example:

Play order:
Wasabi, Nigiri-3 → 9 points
Wasabi, Nigiri-2, Nigiri-3 → 6 + 3
Nigiri-3, Wasabi → 3 points only

---

## 5.6 Maki

Each Maki card contributes icons:

Maki-1 → 1 icon  
Maki-2 → 2 icons  
Maki-3 → 3 icons  

Total maki icons are summed per player.

Scoring (2 players only):

- Player with higher total receives 6 points.
- Player with lower total receives 3 points.
- If totals are equal:
    - Split 6 + 3 = 9 points evenly.
    - Each player receives 4.5 points.

Scoring MUST allow fractional values.

---

# 6. Total Score

Each player's total score is:

Tempura
+ Sashimi
+ Dumpling
+ Nigiri (including Wasabi effects)
+ Maki bonus

Scores may be float due to maki ties.

---

# 7. Reward Function

Reward is ONLY given at terminal state.

Reward = my_total_score − opponent_total_score

Intermediate rewards = 0

---

# 8. Episode Termination

Episode terminates exactly after HAND_SIZE turns.

There is no truncation except if externally forced.

---

# 9. Observation Requirements (Implementation Constraint)

The environment must:

- Provide fixed-size observation vector.
- Provide action mask.
- Expose enough information to reconstruct:
    - My played pile
    - Opponent played pile
    - Current hand
    - Turn number

Exact encoding is defined in environment code, not here.

---

# 10. Invariants (Must Hold)

- Same seed → same deck → same initial hands.
- Same sequence of actions → identical final score.
- Final score from environment MUST equal direct scoring from played piles.
- Episode length MUST equal HAND_SIZE.

---

# 11. Explicitly Excluded Rules

The following are NOT included:

- Pudding
- Chopsticks
- Multiple rounds
- 3+ players
- Special variants

---

# 12. Scope Control

This specification is frozen for version 1.
No rules may be altered during RL development.

All implementations must strictly follow this file.
