"""Evaluate Sushi Go policies and diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sushigo_rl import rules
from sushigo_rl.agents.heuristic_agent import HeuristicAgent
from sushigo_rl.agents.random_agent import RandomAgent
from sushigo_rl.env import OpponentPolicy, PolicyInput, SushiGoEnv


EPS = 1e-9

PlayerActionFn = Callable[[np.ndarray, np.ndarray, PolicyInput], int]


@dataclass(frozen=True)
class EvalSummary:
    """Aggregated matchup metrics."""

    label: str
    episodes: int
    wins: int
    losses: int
    ties: int
    tie_rate: float
    win_rate_including_ties: float
    win_rate_excluding_ties: float
    mean_score_diff: float
    std_score_diff: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    mean_my_score: float
    mean_opp_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a MaskablePPO Sushi Go model")
    parser.add_argument("--model", type=Path, default=None, help="Path to model (.zip or base name)")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per matchup")
    parser.add_argument("--seed", type=int, default=123, help="Evaluation seed base")
    parser.add_argument("--hand-size", type=int, default=rules.HAND_SIZE, help="Environment hand size")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy actions (default: stochastic PPO sampling)",
    )
    parser.add_argument(
        "--opponents",
        choices=("none", "random", "heuristic", "both"),
        default="both",
        help="Model-vs-opponent matchups to run",
    )
    parser.add_argument(
        "--baselines",
        choices=("none", "random_vs_random", "heuristic_vs_random", "policy_vs_self", "both"),
        default="none",
        help="Additional baseline matchups",
    )
    parser.add_argument(
        "--vecnorm-path",
        type=Path,
        default=None,
        help="Path to VecNormalize stats (.pkl). Required if model was trained with --vecnorm",
    )
    parser.add_argument(
        "--fixed-episode-seed",
        type=int,
        default=None,
        help="If set, all evaluation episodes use the same deck/opponent RNG seed",
    )
    parser.add_argument(
        "--env-debug-obs",
        action="store_true",
        help="Enable env observation debug prints (first 5 episodes)",
    )
    parser.add_argument(
        "--repro-check",
        action="store_true",
        help="Run deterministic trajectory reproducibility sanity check",
    )
    parser.add_argument("--repro-seed", type=int, default=17, help="Seed used for reproducibility check")
    parser.add_argument(
        "--repro-actions",
        type=str,
        default=None,
        help="Comma-separated fixed action sequence for reproducibility check",
    )
    return parser.parse_args()


def _load_vecnormalize(vecnorm_path: Path, hand_size: int) -> VecNormalize:
    dummy_env = DummyVecEnv([lambda: SushiGoEnv(hand_size=hand_size)])
    vecnorm = VecNormalize.load(str(vecnorm_path), dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm


def _normalize_obs(vecnorm: VecNormalize | None, obs: np.ndarray) -> np.ndarray:
    if vecnorm is None:
        return obs
    return vecnorm.normalize_obs(obs[np.newaxis, :])[0]


def _build_summary(label: str, my_scores: np.ndarray, opp_scores: np.ndarray) -> EvalSummary:
    score_diffs = my_scores - opp_scores
    wins = int(np.sum(score_diffs > EPS))
    losses = int(np.sum(score_diffs < -EPS))
    ties = int(score_diffs.size - wins - losses)

    win_rate_including_ties = wins / float(score_diffs.size)
    decisive = wins + losses
    win_rate_excluding_ties = float(wins / decisive) if decisive > 0 else float("nan")
    tie_rate = ties / float(score_diffs.size)

    p05, p25, p50, p75, p95 = np.percentile(score_diffs, [5, 25, 50, 75, 95])
    return EvalSummary(
        label=label,
        episodes=int(score_diffs.size),
        wins=wins,
        losses=losses,
        ties=ties,
        tie_rate=tie_rate,
        win_rate_including_ties=win_rate_including_ties,
        win_rate_excluding_ties=win_rate_excluding_ties,
        mean_score_diff=float(np.mean(score_diffs)),
        std_score_diff=float(np.std(score_diffs)),
        p05=float(p05),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        p95=float(p95),
        mean_my_score=float(np.mean(my_scores)),
        mean_opp_score=float(np.mean(opp_scores)),
    )


def _print_summary(summary: EvalSummary) -> None:
    print(f"{summary.label}:")
    print(f"  episodes={summary.episodes}")
    print(f"  wins={summary.wins}, losses={summary.losses}, ties={summary.ties}")
    print(f"  tie_rate={summary.tie_rate:.3f}")
    print(f"  win_rate_including_ties={summary.win_rate_including_ties:.3f}")
    print(f"  win_rate_excluding_ties={summary.win_rate_excluding_ties:.3f}")
    print(f"  score_diff_mean={summary.mean_score_diff:.3f}")
    print(f"  score_diff_std={summary.std_score_diff:.3f}")
    print(
        "  score_diff_percentiles="
        f"p05={summary.p05:.3f}, p25={summary.p25:.3f}, p50={summary.p50:.3f}, "
        f"p75={summary.p75:.3f}, p95={summary.p95:.3f}"
    )
    print(f"  mean_my_score={summary.mean_my_score:.3f}, mean_opp_score={summary.mean_opp_score:.3f}")


def _make_model_player_action(
    model: MaskablePPO,
    deterministic: bool,
    vecnorm: VecNormalize | None,
) -> PlayerActionFn:
    def _select_action(obs: np.ndarray, action_mask: np.ndarray, _: PolicyInput) -> int:
        obs_input = _normalize_obs(vecnorm, obs)
        action, _ = model.predict(obs_input, deterministic=deterministic, action_masks=action_mask)
        return int(action)

    return _select_action


def _make_model_opponent_policy(
    model: MaskablePPO,
    deterministic: bool,
    vecnorm: VecNormalize | None,
    hand_size: int,
) -> OpponentPolicy:
    def _policy(policy_input) -> int:
        opp_obs = SushiGoEnv.encode_observation(
            my_played=policy_input.my_played,
            opp_played=policy_input.opp_played,
            my_hand=policy_input.hand,
            turn=policy_input.turn,
            current_hand_size=policy_input.hand_size,
            hand_size=hand_size,
        )
        opp_obs_input = _normalize_obs(vecnorm, opp_obs)
        action, _ = model.predict(
            opp_obs_input,
            deterministic=deterministic,
            action_masks=policy_input.action_mask,
        )
        return int(action)

    return _policy


def _make_random_player_action(seed: int) -> PlayerActionFn:
    rng = np.random.default_rng(seed)

    def _select_action(_: np.ndarray, action_mask: np.ndarray, __: PolicyInput) -> int:
        legal = np.flatnonzero(action_mask)
        if legal.size == 0:
            raise ValueError("No legal actions available for player")
        return int(rng.choice(legal))

    return _select_action


def _make_heuristic_player_action() -> PlayerActionFn:
    heuristic_agent = HeuristicAgent()

    def _select_action(_: np.ndarray, __: np.ndarray, policy_input: PolicyInput) -> int:
        return int(heuristic_agent.select_action(policy_input))

    return _select_action


def run_matchup(
    label: str,
    episodes: int,
    seed: int,
    hand_size: int,
    player_action_fn: PlayerActionFn,
    opponent_policy: OpponentPolicy,
    fixed_episode_seed: int | None,
    env_debug_obs: bool,
) -> EvalSummary:
    """Evaluate one matchup and return full diagnostics."""
    env = SushiGoEnv(
        hand_size=hand_size,
        opponent_policy=opponent_policy,
        fixed_episode_seed=fixed_episode_seed,
        env_debug_obs=env_debug_obs,
    )

    my_scores = np.zeros(episodes, dtype=np.float64)
    opp_scores = np.zeros(episodes, dtype=np.float64)

    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        terminated = False
        truncated = False
        reward = 0.0
        info: dict[str, object] = {}

        while not (terminated or truncated):
            action_mask = env.action_masks()
            policy_input = env._policy_input_for_player(player_index=0, mask=action_mask)
            action = player_action_fn(obs, action_mask, policy_input)
            obs, reward, terminated, truncated, info = env.step(action)

        my_score = float(info["my_score"])
        opp_score = float(info["opp_score"])
        if not np.isclose(reward, my_score - opp_score):
            raise AssertionError(
                f"Terminal reward mismatch: reward={reward}, expected={my_score - opp_score}"
            )

        my_scores[episode_idx] = my_score
        opp_scores[episode_idx] = opp_score

    env.close()
    return _build_summary(label=label, my_scores=my_scores, opp_scores=opp_scores)


def _parse_action_sequence(spec: str | None, hand_size: int) -> list[int]:
    if spec is None:
        return [0] * hand_size

    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts:
        raise ValueError("--repro-actions cannot be empty")

    actions = [int(part) for part in parts]
    if len(actions) < hand_size:
        actions.extend([0] * (hand_size - len(actions)))
    return actions[:hand_size]


def _run_fixed_trajectory(seed: int, hand_size: int, actions: list[int]) -> list[dict[str, object]]:
    env = SushiGoEnv(hand_size=hand_size)
    obs, _ = env.reset(seed=seed)
    trajectory: list[dict[str, object]] = []

    terminated = False
    truncated = False
    step_idx = 0

    while not (terminated or truncated):
        action_mask = env.action_masks()
        action = actions[step_idx]
        if action < 0 or action >= len(action_mask) or not bool(action_mask[action]):
            raise ValueError(f"Repro action {action} is illegal at step {step_idx} with mask={action_mask.tolist()}")

        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(
            {
                "obs": obs.copy(),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "turn": int(info["turn"]),
                "my_score": float(info["my_score"]),
                "opp_score": float(info["opp_score"]),
                "my_played": tuple(info["my_played"]),
                "opp_played": tuple(info["opp_played"]),
                "my_hand": tuple(info["my_hand"]),
                "opp_hand": tuple(info["opp_hand"]),
                "action_mask": tuple(bool(v) for v in info["action_mask"].tolist()),
                "last_actions": info["last_actions"],
            }
        )
        step_idx += 1

    env.close()
    return trajectory


def run_reproducibility_sanity_check(seed: int, hand_size: int, actions: list[int]) -> None:
    """Assert deterministic trajectories for identical seed + action sequence."""
    traj_a = _run_fixed_trajectory(seed=seed, hand_size=hand_size, actions=actions)
    traj_b = _run_fixed_trajectory(seed=seed, hand_size=hand_size, actions=actions)

    if len(traj_a) != len(traj_b):
        raise AssertionError("Trajectory lengths differ across repeated runs")

    for idx, (step_a, step_b) in enumerate(zip(traj_a, traj_b)):
        if not np.array_equal(step_a["obs"], step_b["obs"]):
            raise AssertionError(f"Observation mismatch at step {idx}")

        for key in (
            "reward",
            "terminated",
            "truncated",
            "turn",
            "my_score",
            "opp_score",
            "my_played",
            "opp_played",
            "my_hand",
            "opp_hand",
            "action_mask",
            "last_actions",
        ):
            if step_a[key] != step_b[key]:
                raise AssertionError(f"Trajectory mismatch at step {idx} for key '{key}'")

    terminal = traj_a[-1]
    expected = float(terminal["my_score"]) - float(terminal["opp_score"])
    if not np.isclose(float(terminal["reward"]), expected):
        raise AssertionError(
            f"Terminal reward mismatch in reproducibility check: reward={terminal['reward']}, expected={expected}"
        )

    print(
        "repro_check: passed "
        f"(seed={seed}, steps={len(traj_a)}, final_my_score={terminal['my_score']}, "
        f"final_opp_score={terminal['opp_score']}, terminal_reward={terminal['reward']})"
    )


def main() -> None:
    args = parse_args()

    needs_model = args.opponents != "none" or args.baselines in {"policy_vs_self", "both"}
    if needs_model and args.model is None:
        raise ValueError("--model is required for model-based evaluations")

    model: MaskablePPO | None = None
    vecnorm: VecNormalize | None = None
    if args.model is not None:
        model_path = args.model.with_suffix("") if args.model.suffix == ".zip" else args.model
        model = MaskablePPO.load(str(model_path))
        if args.vecnorm_path is not None:
            vecnorm = _load_vecnormalize(args.vecnorm_path, args.hand_size)

    if args.repro_check:
        action_sequence = _parse_action_sequence(args.repro_actions, args.hand_size)
        run_reproducibility_sanity_check(seed=args.repro_seed, hand_size=args.hand_size, actions=action_sequence)

    if args.opponents in {"random", "both"}:
        assert model is not None
        random_opponent = RandomAgent(seed=args.seed + 7)
        summary = run_matchup(
            label="policy vs random",
            episodes=args.episodes,
            seed=args.seed,
            hand_size=args.hand_size,
            player_action_fn=_make_model_player_action(model, args.deterministic, vecnorm),
            opponent_policy=random_opponent.select_action,
            fixed_episode_seed=args.fixed_episode_seed,
            env_debug_obs=args.env_debug_obs,
        )
        _print_summary(summary)

    if args.opponents in {"heuristic", "both"}:
        assert model is not None
        heuristic_opponent = HeuristicAgent()
        summary = run_matchup(
            label="policy vs heuristic",
            episodes=args.episodes,
            seed=args.seed + 10_000,
            hand_size=args.hand_size,
            player_action_fn=_make_model_player_action(model, args.deterministic, vecnorm),
            opponent_policy=heuristic_opponent.select_action,
            fixed_episode_seed=args.fixed_episode_seed,
            env_debug_obs=args.env_debug_obs,
        )
        _print_summary(summary)

    if args.baselines in {"random_vs_random", "both"}:
        random_opponent = RandomAgent(seed=args.seed + 101)
        summary = run_matchup(
            label="random vs random",
            episodes=args.episodes,
            seed=args.seed + 20_000,
            hand_size=args.hand_size,
            player_action_fn=_make_random_player_action(seed=args.seed + 202),
            opponent_policy=random_opponent.select_action,
            fixed_episode_seed=args.fixed_episode_seed,
            env_debug_obs=args.env_debug_obs,
        )
        _print_summary(summary)

    if args.baselines in {"heuristic_vs_random", "both"}:
        random_opponent = RandomAgent(seed=args.seed + 401)
        summary = run_matchup(
            label="heuristic vs random",
            episodes=args.episodes,
            seed=args.seed + 30_000,
            hand_size=args.hand_size,
            player_action_fn=_make_heuristic_player_action(),
            opponent_policy=random_opponent.select_action,
            fixed_episode_seed=args.fixed_episode_seed,
            env_debug_obs=args.env_debug_obs,
        )
        _print_summary(summary)

    if args.baselines in {"policy_vs_self", "both"}:
        assert model is not None
        summary = run_matchup(
            label="policy vs itself",
            episodes=args.episodes,
            seed=args.seed + 40_000,
            hand_size=args.hand_size,
            player_action_fn=_make_model_player_action(model, args.deterministic, vecnorm),
            opponent_policy=_make_model_opponent_policy(model, args.deterministic, vecnorm, args.hand_size),
            fixed_episode_seed=args.fixed_episode_seed,
            env_debug_obs=args.env_debug_obs,
        )
        _print_summary(summary)

    if vecnorm is not None:
        vecnorm.close()


if __name__ == "__main__":
    main()
