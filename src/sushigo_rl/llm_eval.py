"""Run quantitative evaluation of LLM explain/coach outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from sushigo_rl import rules
from sushigo_rl.agents.random_agent import RandomAgent
from sushigo_rl.env import SushiGoEnv
from sushigo_rl.llm_assistant import LLMAssistant, PolicyAdvisor, evaluate_explanations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM explanation quality metrics")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/final_policy.zip"))
    parser.add_argument("--vecnorm-path", type=Path, default=Path("artifacts/final_vecnorm.pkl"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=("explain", "coach", "both"),
        default="both",
        help="Evaluation mode(s) to run",
    )
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--no-llm", action="store_true", help="Force fallback templates")
    parser.add_argument(
        "--provider",
        choices=("auto", "openai", "gemini", "fallback"),
        default="auto",
        help="LLM provider selection",
    )
    parser.add_argument("--csv-path", type=Path, default=Path("runs/llm_logs/llm_evaluation.csv"))
    return parser.parse_args()


def _print_metrics(metrics: dict[str, object]) -> None:
    print(
        f"mode={metrics['mode']} "
        f"episodes={metrics['episodes']} "
        f"steps={metrics['steps']} "
        f"avg_words={float(metrics['avg_words']):.3f} "
        f"key_term_rate={float(metrics['key_term_rate']):.3f} "
        f"distinct_ratio={float(metrics['distinct_ratio']):.3f} "
        f"provider={metrics['provider']} "
        f"fallback_mode={metrics['fallback_mode']} "
        f"model_name={metrics['model_name']}"
    )


def main() -> None:
    args = parse_args()
    vecnorm_path = args.vecnorm_path if args.vecnorm_path.exists() else None
    advisor = PolicyAdvisor(model_path=args.model_path, hand_size=rules.HAND_SIZE, vecnorm_path=vecnorm_path)
    assistant = LLMAssistant(
        policy_advisor=advisor,
        model_name=args.llm_model,
        api_key="" if args.no_llm else None,
        provider_choice="fallback" if args.no_llm else args.provider,
    )
    print(
        f"selected_provider={assistant.provider_name} "
        f"model={assistant.model_name} "
        f"fallback_mode={assistant.fallback_mode}"
    )
    random_opponent = RandomAgent(seed=args.seed + 1000)
    env = SushiGoEnv(hand_size=rules.HAND_SIZE, opponent_policy=random_opponent.select_action)

    modes = ["explain", "coach"] if args.mode == "both" else [args.mode]
    for mode in modes:
        metrics = evaluate_explanations(
            policy=advisor,
            env=env,
            num_episodes=args.episodes,
            mode=mode,
            assistant=assistant,
            topk=args.topk,
            seed=args.seed,
            csv_path=args.csv_path,
        )
        _print_metrics(metrics)

    env.close()
    print(f"CSV updated: {args.csv_path}")


if __name__ == "__main__":
    main()
