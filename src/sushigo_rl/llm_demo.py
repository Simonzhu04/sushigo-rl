"""Run small LLM explanation demos and save plain-text transcripts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from sushigo_rl import rules
from sushigo_rl.agents.random_agent import RandomAgent
from sushigo_rl.env import SushiGoEnv
from sushigo_rl.llm_assistant import LLMAssistant, PolicyAdvisor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM demo transcripts")
    parser.add_argument("--model-path", type=Path, default=Path("runs/guanfang_best_20260316.zip"))
    parser.add_argument("--vecnorm-path", type=Path, default=Path("runs/guanfang_best_20260316.vecnormalize.pkl"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument("--no-llm", action="store_true", help="Force fallback templates")
    parser.add_argument(
        "--provider",
        choices=("auto", "openai", "gemini", "fallback"),
        default="auto",
        help="LLM provider selection",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


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

    random_opponent = RandomAgent(seed=args.seed + 500)
    env = SushiGoEnv(hand_size=rules.HAND_SIZE, opponent_policy=random_opponent.select_action)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output_path or Path(f"runs/llm_logs/llm_demo_{timestamp}.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        lines.append(line)

    emit(f"llm_demo_start timestamp={timestamp} episodes={args.episodes}")
    emit(
        f"llm_provider={assistant.provider_name} "
        f"llm_mode={'fallback' if assistant.fallback_mode else 'api'} "
        f"model={assistant.model_name}"
    )
    emit(f"assistant_log={assistant.log_path}")

    for episode_idx in range(args.episodes):
        emit("")
        emit(f"=== Episode {episode_idx + 1}/{args.episodes} ===")
        obs, info = env.reset(seed=args.seed + episode_idx)
        terminated = False
        truncated = False
        step_idx = 0

        while not (terminated or truncated):
            summary = assistant.build_state_summary(env=env, info=info, topk=args.topk)
            action = advisor.predict_action(obs=obs, action_mask=env.action_masks(), deterministic=True)
            explanation = assistant.explain_agent_move(summary, chosen_action=action, topk=args.topk)

            emit(f"step={step_idx + 1}")
            emit("state_summary_json=" + json.dumps(summary, sort_keys=True))
            emit(f"agent_action={action}")
            emit("llm_explanation=" + explanation)
            if step_idx == 0:
                coach = assistant.coach_user(summary, topk=args.topk)
                emit("llm_coach_first_turn=" + coach)

            obs, reward, terminated, truncated, info = env.step(action)
            step_idx += 1
            if terminated or truncated:
                emit(
                    "episode_result="
                    + json.dumps(
                        {
                            "my_score": float(info["my_score"]),
                            "opp_score": float(info["opp_score"]),
                            "score_diff": float(info["my_score"] - info["opp_score"]),
                            "terminal_reward": float(reward),
                        },
                        sort_keys=True,
                    )
                )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env.close()
    print(f"Saved transcript: {output_path}")


if __name__ == "__main__":
    main()
