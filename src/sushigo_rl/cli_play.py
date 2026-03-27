"""Interactive CLI: human plays Sushi Go against RL or heuristic opponent."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sushigo_rl import rules
from sushigo_rl.agents.heuristic_agent import HeuristicAgent
from sushigo_rl.env import PolicyInput, SushiGoEnv
from sushigo_rl.llm_assistant import LLMAssistant, PolicyAdvisor


@dataclass
class OpponentTrace:
    """Stores opponent decision context for post-turn explanation."""

    last_summary: dict[str, Any] | None = None
    last_action: int | None = None


class RLOpponentController:
    """Environment opponent policy backed by frozen RL model."""

    def __init__(self, assistant: LLMAssistant, hand_size: int, topk: int, deterministic: bool = True) -> None:
        if assistant.policy_advisor is None:
            raise ValueError("RLOpponentController requires assistant with PolicyAdvisor")
        self.assistant = assistant
        self.hand_size = hand_size
        self.topk = topk
        self.deterministic = deterministic
        self.trace = OpponentTrace()

    def __call__(self, policy_input: PolicyInput) -> int:
        summary = self.assistant.build_policy_input_summary(policy_input, hand_size=self.hand_size, topk=self.topk)
        action = self.assistant.policy_advisor.predict_action_for_policy_input(
            policy_input=policy_input,
            deterministic=self.deterministic,
        )
        self.trace.last_summary = summary
        self.trace.last_action = int(action)
        return int(action)


class HeuristicOpponentController:
    """Environment opponent policy backed by deterministic heuristic."""

    def __init__(self, assistant: LLMAssistant, hand_size: int, topk: int) -> None:
        self.assistant = assistant
        self.hand_size = hand_size
        self.topk = topk
        self.heuristic = HeuristicAgent()
        self.trace = OpponentTrace()

    def __call__(self, policy_input: PolicyInput) -> int:
        summary = self.assistant.build_policy_input_summary(policy_input, hand_size=self.hand_size, topk=self.topk)
        action = int(self.heuristic.select_action(policy_input))
        self.trace.last_summary = summary
        self.trace.last_action = action
        return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Sushi Go in CLI against RL or heuristic opponent")
    parser.add_argument(
        "--opponent",
        choices=("rl", "heuristic"),
        default="rl",
        help="Opponent type",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("runs/guanfang_best_20260316.zip"),
        help="Frozen RL policy path (.zip or base path)",
    )
    parser.add_argument(
        "--vecnorm-path",
        type=Path,
        default=Path("runs/guanfang_best_20260316.vecnormalize.pkl"),
        help="VecNormalize stats path (optional)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Episode seed")
    parser.add_argument("--topk", type=int, default=3, help="Top-K moves to show in coach mode")
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Optional override for OPENAI_MODEL",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force deterministic fallback templates even if OPENAI_API_KEY is set",
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Non-interactive mode: print only LLM coach/explain outputs and auto-play top move",
    )
    return parser.parse_args()


def _format_played_counts(cards: tuple[str, ...]) -> str:
    counts = rules.count_cards(cards)
    parts = [f"{card}:{counts[card]}" for card in rules.CARD_TYPES if counts[card] > 0]
    return ", ".join(parts) if parts else "(none)"


def _print_turn_header(info: dict[str, Any], hand_size: int) -> None:
    turn = int(info["turn"])
    round_idx = int(info.get("round_idx", 0))
    num_rounds = int(info.get("num_rounds", 1))
    print(f"\nRound {round_idx + 1}/{num_rounds}, turn {turn + 1}/{hand_size}")
    my_hand = list(info["my_hand"])
    print("Your hand:")
    for idx, card in enumerate(my_hand):
        print(f"  [{idx}] {card}")
    print("Legal actions:")
    legal_actions = np.flatnonzero(info["action_mask"]).tolist()
    for action in legal_actions:
        print(f"  {SushiGoEnv.describe_action(int(action), my_hand, hand_size=hand_size)}")


def _print_visible_state(info: dict[str, Any]) -> None:
    my_played = tuple(info["my_played"])
    opp_played = tuple(info["opp_played"])
    print("Visible played piles:")
    print(f"  You:      {_format_played_counts(my_played)}")
    print(f"  Opponent: {_format_played_counts(opp_played)}")
    print(
        "Maki totals: "
        f"you={rules.count_maki_icons(my_played)} "
        f"opponent={rules.count_maki_icons(opp_played)}"
    )
    print(f"Pudding totals: you={info.get('my_pudding', 0)} opponent={info.get('opp_pudding', 0)}")


def _print_final_breakdown(info: dict[str, Any]) -> None:
    my_score = float(info["my_score"])
    opp_score = float(info["opp_score"])
    print("\nFinal scores:")
    print(f"  You:      {my_score:.1f}")
    print(f"  Opponent: {opp_score:.1f}")
    print(f"  Diff:     {my_score - opp_score:+.1f}")

    my_breakdown = info.get("my_breakdown")
    opp_breakdown = info.get("opp_breakdown")
    if my_breakdown is not None and opp_breakdown is not None:
        print("Score breakdown:")
        print(
            "  You: "
            f"tempura={my_breakdown.tempura:.1f}, "
            f"sashimi={my_breakdown.sashimi:.1f}, "
            f"dumpling={my_breakdown.dumpling:.1f}, "
            f"nigiri={my_breakdown.nigiri:.1f}, "
            f"maki={my_breakdown.maki:.1f}"
        )
        print(
            "  Opponent: "
            f"tempura={opp_breakdown.tempura:.1f}, "
            f"sashimi={opp_breakdown.sashimi:.1f}, "
            f"dumpling={opp_breakdown.dumpling:.1f}, "
            f"nigiri={opp_breakdown.nigiri:.1f}, "
            f"maki={opp_breakdown.maki:.1f}"
        )


def main() -> None:
    args = parse_args()
    vecnorm_path = args.vecnorm_path if args.vecnorm_path.exists() else None

    advisor = PolicyAdvisor(
        model_path=args.model_path,
        hand_size=rules.HAND_SIZE,
        vecnorm_path=vecnorm_path,
    )
    assistant = LLMAssistant(
        policy_advisor=advisor,
        model_name=args.llm_model,
        api_key="" if args.no_llm else None,
    )
    if assistant.fallback_mode:
        print("LLM mode: fallback templates (no OPENAI_API_KEY or no SDK client).")
    else:
        print(f"LLM mode: external API ({assistant.model_name})")
    print(f"Assistant logs: {assistant.log_path}")

    if args.opponent == "rl":
        opponent_controller = RLOpponentController(
            assistant=assistant,
            hand_size=rules.HAND_SIZE,
            topk=args.topk,
            deterministic=True,
        )
    else:
        opponent_controller = HeuristicOpponentController(
            assistant=assistant,
            hand_size=rules.HAND_SIZE,
            topk=args.topk,
        )

    env = SushiGoEnv(hand_size=rules.HAND_SIZE, opponent_policy=opponent_controller)
    obs, info = env.reset(seed=args.seed)
    del obs

    terminated = False
    truncated = False

    while not (terminated or truncated):
        if args.llm_only:
            summary = assistant.build_state_summary(env=env, info=info, topk=args.topk)
            coach_text = assistant.coach_user(summary, topk=args.topk)
            print("\n[llm-only] coach:")
            print(coach_text)
            recommendations = summary.get("agent_recommendation", [])
            if recommendations:
                action = int(recommendations[0]["action_index"])
            else:
                legal = np.flatnonzero(np.asarray(info["action_mask"], dtype=np.bool_))
                action = int(legal[0])
            print(f"[llm-only] auto action: {action}")
        else:
            _print_turn_header(info, env.hand_size)
            _print_visible_state(info)

            while True:
                user_text = input("Choose move [index/help/why/quit]: ").strip().lower()
                if user_text in {"quit", "exit"}:
                    print("Exiting game.")
                    env.close()
                    return

                if user_text == "help":
                    summary = assistant.build_state_summary(env=env, info=info, topk=args.topk)
                    print(assistant.coach_user(summary, topk=args.topk))
                    continue

                if user_text == "why":
                    trace = opponent_controller.trace
                    if trace.last_summary is None or trace.last_action is None:
                        print("No opponent move yet. Play one turn first.")
                        continue
                    print(
                        assistant.explain_agent_move(
                            state_summary=trace.last_summary,
                            chosen_action=trace.last_action,
                            topk=args.topk,
                        )
                    )
                    continue

                try:
                    action = int(user_text)
                except ValueError:
                    print("Invalid input. Use an index, 'help', 'why', or 'quit'.")
                    continue

                mask = np.asarray(info["action_mask"], dtype=np.bool_)
                if action < 0 or action >= len(mask) or not bool(mask[action]):
                    print(f"Illegal action index {action}. Legal: {np.flatnonzero(mask).tolist()}")
                    continue
                break

        obs, reward, terminated, truncated, info = env.step(action)
        del obs

        my_idx, opp_idx = info["last_actions"]
        my_desc, opp_desc = info.get(
            "last_action_descriptions",
            (
                SushiGoEnv.describe_action(my_idx, info["my_hand"], hand_size=env.hand_size),
                SushiGoEnv.describe_action(opp_idx, info["opp_hand"], hand_size=env.hand_size),
            ),
        )
        print(
            f"Turn result: you played {my_desc}; "
            f"opponent played {opp_desc}"
        )
        if args.llm_only:
            trace = opponent_controller.trace
            if trace.last_summary is not None and trace.last_action is not None:
                print("[llm-only] explain opponent:")
                print(
                    assistant.explain_agent_move(
                        state_summary=trace.last_summary,
                        chosen_action=trace.last_action,
                        topk=args.topk,
                    )
                )
        if terminated:
            print(f"Terminal reward (score diff): {reward:+.1f}")

    _print_final_breakdown(info)

    request_post_game = True
    if not args.llm_only:
        post_game = input("Request post-game summary? [y/N]: ").strip().lower()
        request_post_game = post_game in {"y", "yes"}

    if request_post_game:
        summary = assistant.build_state_summary(env=env, info=info, topk=args.topk)
        text = assistant.summarize_post_game(
            state_summary=summary,
            my_score=float(info["my_score"]),
            opp_score=float(info["opp_score"]),
            my_breakdown=info.get("my_breakdown"),
            opp_breakdown=info.get("opp_breakdown"),
        )
        print(text)

    env.close()


if __name__ == "__main__":
    main()
