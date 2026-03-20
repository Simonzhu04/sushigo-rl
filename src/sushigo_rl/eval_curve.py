"""Evaluate checkpoint learning curves and export CSV/PNG artifacts."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize

from sushigo_rl import rules
from sushigo_rl.agents.heuristic_agent import HeuristicAgent
from sushigo_rl.agents.random_agent import RandomAgent
from sushigo_rl.eval import EvalSummary, _load_vecnormalize, run_matchup
from sushigo_rl.env import OpponentPolicy, PolicyInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints and plot learning curves")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Directory containing checkpoint .zip files")
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="model",
        help="Checkpoint filename prefix (for example: model_100000_steps.zip)",
    )
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per checkpoint matchup")
    parser.add_argument("--seed", type=int, default=123, help="Base evaluation seed")
    parser.add_argument("--hand-size", type=int, default=rules.HAND_SIZE, help="Environment hand size")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic model actions")
    parser.add_argument(
        "--opponents",
        choices=("random", "heuristic", "both"),
        default="both",
        help="Opponent matchups to evaluate per checkpoint",
    )
    parser.add_argument(
        "--vecnorm-dir",
        type=Path,
        default=None,
        help="Directory containing per-checkpoint VecNormalize stats",
    )
    parser.add_argument(
        "--vecnorm-path",
        type=Path,
        default=None,
        help="Single VecNormalize stats path to use for all checkpoints",
    )
    parser.add_argument(
        "--fixed-episode-seed",
        type=int,
        default=None,
        help="If set, all evaluation episodes use the same env seed",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Output CSV path (default: <checkpoint-dir>/learning_curve.csv)",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Output PNG path (default: <checkpoint-dir>/learning_curve.png)",
    )
    return parser.parse_args()


def _extract_timesteps(path: Path, prefix: str) -> int | None:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_steps\.zip$")
    match = pattern.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _discover_checkpoints(checkpoint_dir: Path, prefix: str) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob(f"{prefix}_*_steps.zip"):
        timesteps = _extract_timesteps(path, prefix)
        if timesteps is None:
            continue
        checkpoints.append((timesteps, path))
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints


def _resolve_vecnorm_for_checkpoint(
    timesteps: int,
    args: argparse.Namespace,
) -> Path | None:
    if args.vecnorm_path is not None:
        return args.vecnorm_path

    if args.vecnorm_dir is None:
        return None

    prefix = args.checkpoint_prefix
    candidates = [
        args.vecnorm_dir / f"{prefix}_vecnormalize_{timesteps}_steps.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _make_model_player_action(
    model: MaskablePPO,
    deterministic: bool,
    vecnorm: VecNormalize | None,
):
    def _select_action(obs: np.ndarray, action_mask: np.ndarray, _: PolicyInput) -> int:
        obs_input = obs
        if vecnorm is not None:
            obs_input = vecnorm.normalize_obs(obs[np.newaxis, :])[0]
        action, _ = model.predict(obs_input, deterministic=deterministic, action_masks=action_mask)
        return int(action)

    return _select_action


# def _summaries_to_csv_rows(timesteps: int, summaries: list[EvalSummary]) -> list[dict[str, float | int | str]]:
#     rows: list[dict[str, float | int | str]] = []
#     for summary in summaries:
#         opponent = summary.label.replace("policy vs ", "")
#         rows.append(
#             {
#                 "timesteps": timesteps,
#                 "opponent": opponent,
#                 "win_rate_excl_ties": summary.win_rate_excluding_ties,
#                 "win_rate_incl_ties": summary.win_rate_including_ties,
#                 "tie_rate": summary.tie_rate,
#                 "mean_score_diff": summary.mean_score_diff,
#             }
#         )
#     return rows

def _summaries_to_csv_rows(timesteps: int, summaries: list[EvalSummary]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for summary in summaries:
        opponent = summary.label.replace("policy vs ", "")
        rows.append(
            {
                "timesteps": timesteps,
                "opponent": opponent,
                "win_rate_excl_ties": summary.win_rate_excluding_ties,
                "win_rate_incl_ties": summary.win_rate_including_ties,
                "tie_rate": summary.tie_rate,
                "mean_score_diff": summary.mean_score_diff,
                "std_score_diff": summary.std_score_diff,
                "p05_score_diff": summary.p05_score_diff,
                "p25_score_diff": summary.p25_score_diff,
                "p50_score_diff": summary.p50_score_diff,
                "p75_score_diff": summary.p75_score_diff,
                "p95_score_diff": summary.p95_score_diff,
            }
        )
    return rows


def _print_summary(summary: EvalSummary) -> None:
    print(
        f"  {summary.label}: "
        f"win_excl_ties={summary.win_rate_excluding_ties:.3f}, "
        f"win_incl_ties={summary.win_rate_including_ties:.3f}, "
        f"tie_rate={summary.tie_rate:.3f}, "
        f"mean_score_diff={summary.mean_score_diff:.3f}"
    )


def _write_csv(rows: list[dict[str, float | int | str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # fieldnames = [
    #     "timesteps",
    #     "opponent",
    #     "win_rate_excl_ties",
    #     "win_rate_incl_ties",
    #     "tie_rate",
    #     "mean_score_diff",
    # ]
    fieldnames = [
        "timesteps",
        "opponent",
        "win_rate_excl_ties",
        "win_rate_incl_ties",
        "tie_rate",
        "mean_score_diff",
        "std_score_diff",
        "p05_score_diff",
        "p25_score_diff",
        "p50_score_diff",
        "p75_score_diff",
        "p95_score_diff",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_learning_curve(rows: list[dict[str, float | int | str]], plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        opponent = str(row["opponent"])
        grouped.setdefault(opponent, []).append(row)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for opponent, values in sorted(grouped.items()):
        values.sort(key=lambda item: int(item["timesteps"]))
        steps = [int(item["timesteps"]) for item in values]
        win_rates = [float(item["win_rate_excl_ties"]) for item in values]
        mean_diffs = [float(item["mean_score_diff"]) for item in values]
        p25 = [float(item["p25_score_diff"]) for item in values]
        p75 = [float(item["p75_score_diff"]) for item in values]

        ax1.plot(steps, win_rates, marker="o", label=opponent)
        ax2.plot(steps, mean_diffs, marker="o", label=opponent)
        ax2.fill_between(steps, p25, p75, alpha=0.2)

    ax1.set_ylabel("Win Rate (Excl. Ties)")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Mean Score Diff")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("Sushi Go RL Learning Curve", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints = _discover_checkpoints(checkpoint_dir, args.checkpoint_prefix)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir} matching "
            f"'{args.checkpoint_prefix}_*_steps.zip'"
        )

    csv_out = args.csv_out or (checkpoint_dir / "learning_curve.csv")
    plot_out = args.plot_out or (checkpoint_dir / "learning_curve.png")
    csv_rows: list[dict[str, float | int | str]] = []

    random_opponent = RandomAgent(seed=args.seed + 7)
    heuristic_opponent = HeuristicAgent()

    for timesteps, checkpoint_path in checkpoints:
        print(f"Evaluating checkpoint: {checkpoint_path} (timesteps={timesteps})")
        model = MaskablePPO.load(str(checkpoint_path))

        vecnorm = None
        vecnorm_path = _resolve_vecnorm_for_checkpoint(timesteps=timesteps, args=args)
        if vecnorm_path is not None:
            vecnorm = _load_vecnormalize(vecnorm_path, args.hand_size)

        summaries: list[EvalSummary] = []
        if args.opponents in {"random", "both"}:
            summary = run_matchup(
                label="policy vs random",
                episodes=args.episodes,
                seed=args.seed,
                hand_size=args.hand_size,
                player_action_fn=_make_model_player_action(model, args.deterministic, vecnorm),
                opponent_policy=random_opponent.select_action,
                fixed_episode_seed=args.fixed_episode_seed,
                env_debug_obs=False,
            )
            summaries.append(summary)
            _print_summary(summary)

        if args.opponents in {"heuristic", "both"}:
            summary = run_matchup(
                label="policy vs heuristic",
                episodes=args.episodes,
                seed=args.seed + 10_000,
                hand_size=args.hand_size,
                player_action_fn=_make_model_player_action(model, args.deterministic, vecnorm),
                opponent_policy=heuristic_opponent.select_action,
                fixed_episode_seed=args.fixed_episode_seed,
                env_debug_obs=False,
            )
            summaries.append(summary)
            _print_summary(summary)

        csv_rows.extend(_summaries_to_csv_rows(timesteps, summaries))
        if vecnorm is not None:
            vecnorm.close()

    _write_csv(csv_rows, csv_out)
    _plot_learning_curve(csv_rows, plot_out)
    print(f"Wrote CSV: {csv_out}")
    print(f"Wrote plot: {plot_out}")


if __name__ == "__main__":
    main()
