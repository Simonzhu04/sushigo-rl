"""Train a Sushi Go agent with MaskablePPO."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sushigo_rl import rules
from sushigo_rl.agents.heuristic_agent import HeuristicAgent
from sushigo_rl.agents.random_agent import RandomAgent
from sushigo_rl.env import OpponentPolicy, OpponentSampler, SushiGoEnv


@dataclass
class OpponentScheduler:
    """Stateful per-episode opponent selector."""

    mode: str
    mix_random_prob: float
    curriculum_stages: list[tuple[float, int]]
    random_policy: OpponentPolicy
    heuristic_policy: OpponentPolicy
    current_timestep: int = 0

    def set_timestep(self, timestep: int) -> None:
        self.current_timestep = max(0, int(timestep))

    def sample(self, rng: np.random.Generator) -> OpponentPolicy:
        if self.mode == "random":
            return self.random_policy

        if self.mode == "heuristic":
            return self.heuristic_policy

        if self.mode == "mix":
            return self.random_policy if float(rng.random()) < self.mix_random_prob else self.heuristic_policy

        if self.mode == "curriculum":
            heuristic_prob = self._curriculum_heuristic_prob()
            return self.heuristic_policy if float(rng.random()) < heuristic_prob else self.random_policy

        raise ValueError(f"Unknown opponent mode: {self.mode}")

    def _curriculum_heuristic_prob(self) -> float:
        elapsed = self.current_timestep
        cumulative = 0
        for heuristic_prob, stage_timesteps in self.curriculum_stages:
            cumulative += stage_timesteps
            if elapsed < cumulative:
                return heuristic_prob
        return self.curriculum_stages[-1][0]


class TimestepSyncCallback(BaseCallback):
    """Keep opponent scheduler aligned with current PPO timestep count."""

    def __init__(self, scheduler: OpponentScheduler) -> None:
        super().__init__()
        self._scheduler = scheduler

    def _on_training_start(self) -> None:
        self._scheduler.set_timestep(0)

    def _on_step(self) -> bool:
        self._scheduler.set_timestep(self.num_timesteps)
        return True

####
def linear_schedule_with_floor(initial_value: float, floor_ratio: float = 0.2):
    def func(progress_remaining: float) -> float:
        return initial_value * (floor_ratio + (1.0 - floor_ratio) * progress_remaining)
    return func
###

def parse_curriculum_stages(spec: str) -> list[tuple[float, int]]:
    """Parse curriculum string: '0.0:200000,0.2:200000,0.5:200000'."""
    stages: list[tuple[float, int]] = []
    raw_parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not raw_parts:
        raise ValueError("curriculum stages cannot be empty")

    for part in raw_parts:
        if ":" not in part:
            raise ValueError(f"Invalid curriculum stage '{part}'. Expected '<heuristic_prob>:<timesteps>'")
        prob_text, steps_text = part.split(":", maxsplit=1)
        heuristic_prob = float(prob_text)
        stage_timesteps = int(steps_text)
        if not 0.0 <= heuristic_prob <= 1.0:
            raise ValueError(f"Heuristic probability must be in [0, 1], got {heuristic_prob}")
        if stage_timesteps <= 0:
            raise ValueError(f"Stage timesteps must be > 0, got {stage_timesteps}")
        stages.append((heuristic_prob, stage_timesteps))

    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on the 3-round Sushi Go variant")
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=None,
        help="TensorBoard log directory (optional; disabled by default)",
    )
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Global RNG seed")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("runs/latest"),
        help="Model output base path (SB3 adds .zip if missing)",
    )
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="Optional existing model (.zip or base path) to continue training from",
    )
    parser.add_argument(
    "--hand-size",
    type=int,
    default=rules.HAND_SIZE,
    help="Environment hand size (must equal rules.HAND_SIZE when chopsticks action space is enabled)",
    )
    parser.add_argument(
        "--opponent-mode",
        type=str,
        default="mix",
        choices=("random", "heuristic", "mix", "curriculum"),
        help="Opponent selection mode (chosen once per episode at reset)",
    )
    parser.add_argument(
        "--mix-random-prob",
        type=float,
        default=0.5,
        help="Random-opponent probability when --opponent-mode=mix",
    )
    parser.add_argument(
        "--curriculum-stages",
        type=str,
        default="0.0:200000,0.2:200000,0.5:200000",
        help="Curriculum stages as '<heuristic_prob>:<timesteps>' comma list",
    )
    parser.add_argument(
        "--random-only-experiment",
        action="store_true",
        help="Shortcut: train 200k steps vs random only and save to runs/random_only",
    )
    parser.add_argument(
        "--overfit-test",
        action="store_true",
        help="Shortcut: fixed-seed 50k random-only training + same-seed eval",
    )
    parser.add_argument(
        "--overfit-eval-episodes",
        type=int,
        default=200,
        help="Evaluation episodes used by --overfit-test",
    )

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--vecnorm", action="store_true", help="Use VecNormalize for observations")
    parser.add_argument(
        "--vecnorm-norm-reward",
        action="store_true",
        help="Also normalize rewards inside VecNormalize (off by default)",
    )
    parser.add_argument(
        "--vecnorm-path",
        type=Path,
        default=None,
        help="Optional output path for VecNormalize stats",
    )
    parser.add_argument(
        "--env-debug-mask",
        action="store_true",
        help="Enable strict action-mask assertion checks at every step",
    )
    parser.add_argument(
        "--env-debug-obs",
        action="store_true",
        help="Enable observation feature debug prints for first 5 episodes",
    )
    parser.add_argument(
        "--fixed-episode-seed",
        type=int,
        default=None,
        help="If set, every episode uses the same environment seed (no deck variation)",
    )
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="If > 0, save model checkpoints every N timesteps",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional checkpoint output directory",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="model",
        help="Filename prefix for checkpoints",
    )
    return parser.parse_args()


# def build_opponent_scheduler(args: argparse.Namespace) -> OpponentScheduler:
#     if not 0.0 <= args.mix_random_prob <= 1.0:
#         raise ValueError("--mix-random-prob must be in [0, 1]")

#     random_agent = RandomAgent(seed=args.seed + 1)
#     heuristic_agent = HeuristicAgent()

#     return OpponentScheduler(
#         mode=args.opponent_mode,
#         mix_random_prob=args.mix_random_prob,
#         curriculum_stages=parse_curriculum_stages(args.curriculum_stages),
#         random_policy=random_agent.select_action,
#         heuristic_policy=heuristic_agent.select_action,
#     )

def build_opponent_scheduler(args: argparse.Namespace) -> OpponentScheduler:
    if not 0.0 <= args.mix_random_prob <= 1.0:
        raise ValueError("--mix-random-prob must be in [0, 1]")

    random_agent = RandomAgent(seed=args.seed + 1)
    heuristic_agent = HeuristicAgent()

    curriculum_stages: list[tuple[float, int]]
    if args.opponent_mode == "curriculum":
        curriculum_stages = parse_curriculum_stages(args.curriculum_stages)
    else:
        curriculum_stages = [(0.0, max(1, int(args.timesteps)))]

    return OpponentScheduler(
        mode=args.opponent_mode,
        mix_random_prob=args.mix_random_prob,
        curriculum_stages=curriculum_stages,
        random_policy=random_agent.select_action,
        heuristic_policy=heuristic_agent.select_action,
    )


def build_env(args: argparse.Namespace, opponent_sampler: OpponentSampler) -> SushiGoEnv:
    return SushiGoEnv(
        hand_size=args.hand_size,
        opponent_sampler=opponent_sampler,
        fixed_episode_seed=args.fixed_episode_seed,
        env_debug_mask=args.env_debug_mask,
        env_debug_obs=args.env_debug_obs,
    )


def _normalize_eval_obs(vecnorm: VecNormalize | None, obs: np.ndarray) -> np.ndarray:
    if vecnorm is None:
        return obs
    return vecnorm.normalize_obs(obs[np.newaxis, :])[0]


def evaluate_vs_random(
    model: MaskablePPO,
    episodes: int,
    seed: int,
    hand_size: int,
    deterministic: bool,
    fixed_episode_seed: int | None,
    vecnorm: VecNormalize | None,
) -> tuple[float, float]:
    """Return (win_rate_including_ties, mean_score_diff) vs random opponent."""
    random_opponent = RandomAgent(seed=seed + 999)
    env = SushiGoEnv(
        hand_size=hand_size,
        opponent_policy=random_opponent.select_action,
        fixed_episode_seed=fixed_episode_seed,
    )

    wins = 0
    score_diffs: list[float] = []
    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed + episode_idx)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            mask = env.action_masks()
            obs_input = _normalize_eval_obs(vecnorm, obs)
            action, _ = model.predict(obs_input, deterministic=deterministic, action_masks=mask)
            obs, _, terminated, truncated, info = env.step(int(action))

        score_diff = float(info["my_score"]) - float(info["opp_score"])
        score_diffs.append(score_diff)
        if score_diff > 0:
            wins += 1

    env.close()
    return wins / float(episodes), float(np.mean(score_diffs))


def main() -> None:
    args = parse_args()
    if args.hand_size != rules.HAND_SIZE:
        raise ValueError(
            f"--hand-size={args.hand_size} is not supported when using the fixed chopsticks action encoding. "
            f"Please use --hand-size {rules.HAND_SIZE} (rules.HAND_SIZE)."
        )

    if args.random_only_experiment:
        args.opponent_mode = "random"
        args.timesteps = 200_000
        args.model_out = Path("runs/random_only")

    if args.overfit_test:
        args.opponent_mode = "random"
        args.timesteps = 50_000
        args.fixed_episode_seed = args.seed if args.fixed_episode_seed is None else args.fixed_episode_seed
        args.model_out = Path("runs/overfit_fixed_seed")

    scheduler = build_opponent_scheduler(args)

    if args.vecnorm:
        vec_env = DummyVecEnv([lambda: build_env(args, scheduler.sample)])
        if args.init_model is not None and args.vecnorm_path is not None and args.vecnorm_path.exists():
            env = VecNormalize.load(str(args.vecnorm_path), vec_env)
            env.training = True
            env.norm_reward = args.vecnorm_norm_reward
        else:
            env = VecNormalize(vec_env, norm_obs=True, norm_reward=args.vecnorm_norm_reward)
        env.seed(args.seed)
    else:
        env = build_env(args, scheduler.sample)
        env.reset(seed=args.seed)

    if args.init_model is not None:
        model_path = args.init_model.with_suffix("") if args.init_model.suffix == ".zip" else args.init_model
        model = MaskablePPO.load(str(model_path), env=env)
        model.set_env(env)
        model.set_random_seed(args.seed)
        model.tensorboard_log = str(args.tensorboard_log) if args.tensorboard_log is not None else None
    else:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            seed=args.seed,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            learning_rate=linear_schedule_with_floor(args.learning_rate, floor_ratio=0.2),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            verbose=args.verbose,
            tensorboard_log=str(args.tensorboard_log) if args.tensorboard_log is not None else None,
        )

    callbacks: list[BaseCallback] = [TimestepSyncCallback(scheduler)]
    if args.checkpoint_every > 0:
        checkpoint_dir = args.checkpoint_dir or (args.model_out.parent / f"{args.model_out.stem}_checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_every,
                save_path=str(checkpoint_dir),
                name_prefix=args.checkpoint_prefix,
                save_replay_buffer=False,
                save_vecnormalize=args.vecnorm,
            )
        )
        print(
            f"Checkpointing every {args.checkpoint_every} steps to {checkpoint_dir} "
            f"(prefix={args.checkpoint_prefix})"
        )

    callback: BaseCallback = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)
    learn_kwargs = dict(
        total_timesteps=args.timesteps,
        reset_num_timesteps=args.init_model is None,
        progress_bar=False,
        callback=callback,
    )
    if args.tensorboard_log is not None:
        learn_kwargs["tb_log_name"] = args.model_out.stem
    model.learn(**learn_kwargs)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_out))
    print(f"Saved model to {args.model_out}.zip")

    if args.vecnorm:
        vecnorm_path = args.vecnorm_path or args.model_out.with_suffix(".vecnormalize.pkl")
        env.save(str(vecnorm_path))
        print(f"Saved VecNormalize stats to {vecnorm_path}")

    if args.overfit_test:
        eval_vecnorm = env if isinstance(env, VecNormalize) else None
        if eval_vecnorm is not None:
            eval_vecnorm.training = False
            eval_vecnorm.norm_reward = False
        win_rate, mean_score_diff = evaluate_vs_random(
            model=model,
            episodes=args.overfit_eval_episodes,
            seed=args.seed,
            hand_size=args.hand_size,
            deterministic=True,
            fixed_episode_seed=args.fixed_episode_seed,
            vecnorm=eval_vecnorm,
        )
        print(
            "[overfit_test] "
            f"episodes={args.overfit_eval_episodes} "
            f"win_rate_including_ties={win_rate:.3f} "
            f"mean_score_diff={mean_score_diff:.3f}"
        )

    env.close()


if __name__ == "__main__":
    main()
