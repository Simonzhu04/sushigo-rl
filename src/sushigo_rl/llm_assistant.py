"""LLM-backed explain/coach layer for Sushi Go policy interaction."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Sequence

import numpy as np
try:
    import torch
except Exception:
    torch = None

try:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except Exception:
    MaskablePPO = None
    DummyVecEnv = None
    VecNormalize = None

from sushigo_rl import rules
from sushigo_rl.env import PolicyInput, SushiGoEnv
from sushigo_rl.llm_providers import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
    BaseLLMProvider,
    create_llm_provider,
)
STRATEGIC_TERMS: tuple[str, ...] = (
    "pair",
    "triple",
    "maki",
    "wasabi",
    "nigiri",
    "tempura",
    "sashimi",
    "dumpling",
)


class PolicyAdvisor:
    """Inference helper for the frozen RL policy (+ optional VecNormalize)."""

    def __init__(
        self,
        model_path: Path,
        hand_size: int = rules.HAND_SIZE,
        vecnorm_path: Path | None = None,
    ) -> None:
        if MaskablePPO is None or DummyVecEnv is None or VecNormalize is None or torch is None:
            raise ImportError(
                "PolicyAdvisor requires optional training dependencies (torch, sb3-contrib, stable-baselines3)."
            )
        model_base = model_path.with_suffix("") if model_path.suffix == ".zip" else model_path
        self.model = MaskablePPO.load(str(model_base))
        self.hand_size = hand_size
        self.vecnorm = self._load_vecnorm(hand_size, vecnorm_path) if vecnorm_path is not None else None

    @staticmethod
    def _load_vecnorm(hand_size: int, vecnorm_path: Path) -> VecNormalize:
        dummy_env = DummyVecEnv([lambda: SushiGoEnv(hand_size=hand_size)])
        vecnorm = VecNormalize.load(str(vecnorm_path), dummy_env)
        vecnorm.training = False
        vecnorm.norm_reward = False
        return vecnorm

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.vecnorm is None:
            return obs
        return self.vecnorm.normalize_obs(obs[np.newaxis, :])[0]

    def predict_action(self, obs: np.ndarray, action_mask: np.ndarray, deterministic: bool = True) -> int:
        obs_input = self._normalize_obs(obs)
        action, _ = self.model.predict(obs_input, deterministic=deterministic, action_masks=action_mask)
        return int(action)

    def predict_action_for_policy_input(self, policy_input: PolicyInput, deterministic: bool = True) -> int:
        obs = SushiGoEnv.observation_from_policy_input(policy_input, hand_size=self.hand_size)
        return self.predict_action(obs=obs, action_mask=policy_input.action_mask, deterministic=deterministic)

    def action_recommendations(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        hand_cards: Sequence[str],
        topk: int = 3,
    ) -> list[dict[str, Any]]:
        probs = self._action_probabilities(obs=obs, action_mask=action_mask)
        legal_indices = np.flatnonzero(action_mask)
        ranked = sorted(legal_indices.tolist(), key=lambda idx: (-probs[idx], idx))
        top_actions = ranked[: max(0, topk)]
        return [
            {
                "action_index": int(idx),
                "card": self._action_card_label(int(idx), hand_cards),
                "action_label": SushiGoEnv.describe_action(int(idx), hand_cards, hand_size=self.hand_size),
                "probability": float(probs[idx]),
            }
            for idx in top_actions
        ]

    def _action_card_label(self, action_index: int, hand_cards: Sequence[str]) -> str:
        cards = SushiGoEnv.cards_for_action(action_index, hand_cards, hand_size=self.hand_size)
        if len(cards) == 1:
            return cards[0]
        return "+".join(cards)

    def _action_probabilities(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        obs_input = self._normalize_obs(obs).astype(np.float32, copy=False)
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs_input)
        mask_2d = np.asarray(action_mask, dtype=np.bool_)[np.newaxis, :]

        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor, action_masks=mask_2d).distribution
            probs = distribution.probs.detach().cpu().numpy()[0].astype(np.float64, copy=False)

        probs = np.where(action_mask, probs, 0.0)
        total = float(np.sum(probs))
        if total <= 0.0:
            legal = np.flatnonzero(action_mask)
            if legal.size == 0:
                return np.zeros_like(probs, dtype=np.float64)
            uniform = np.zeros_like(probs, dtype=np.float64)
            uniform[legal] = 1.0 / float(legal.size)
            return uniform
        return probs / total


class LLMAssistant:
    """User-facing explain/coach layer with pluggable providers + fallback."""

    def __init__(
        self,
        policy_advisor: PolicyAdvisor | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        provider_choice: str | None = None,
        provider: BaseLLMProvider | None = None,
        prompts_dir: Path | None = None,
        log_dir: Path = Path("runs/llm_logs"),
    ) -> None:
        self.policy_advisor = policy_advisor
        resolved_provider_choice = provider_choice
        if api_key == "" and resolved_provider_choice is None:
            resolved_provider_choice = "fallback"

        self.provider = provider or create_llm_provider(
            provider_choice=resolved_provider_choice,
            openai_api_key=api_key if api_key != "" else None,
            openai_model=model_name or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            gemini_model=model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        )
        self.model_name = self.provider.model_name
        self.provider_name = self.provider.provider_name

        self.prompts_dir = prompts_dir or (Path(__file__).resolve().parents[2] / "prompts")
        self._prompts = {
            "explain_system": self._load_prompt("explain_system.txt", self._default_explain_system_prompt()),
            "explain_user": self._load_prompt("explain_user.txt", self._default_explain_user_prompt()),
            "coach_system": self._load_prompt("coach_system.txt", self._default_coach_system_prompt()),
            "coach_user": self._load_prompt("coach_user.txt", self._default_coach_user_prompt()),
        }
        self._prompt_versions = {
            "explain": self._prompt_version(self._prompts["explain_system"], self._prompts["explain_user"]),
            "coach": self._prompt_version(self._prompts["coach_system"], self._prompts["coach_user"]),
            "post_game": self._prompt_version(self._prompts["coach_system"], self._prompts["coach_user"]),
        }

        log_dir.mkdir(parents=True, exist_ok=True)
        log_name = datetime.now(timezone.utc).strftime("assistant_%Y%m%d_%H%M%S.jsonl")
        self.log_path = log_dir / log_name

    @property
    def fallback_mode(self) -> bool:
        return self.provider.fallback_mode

    def build_state_summary(self, env: SushiGoEnv, info: dict[str, Any], topk: int = 3) -> dict[str, Any]:
        """Build a compact, JSON-serializable summary from env-visible state."""
        my_played = list(info["my_played"])
        opp_played = list(info["opp_played"])
        my_hand = list(info["my_hand"])
        turn = int(info["turn"])
        action_mask = np.asarray(info["action_mask"], dtype=np.bool_)
        return self._summary_from_components(
            my_played=my_played,
            opp_played=opp_played,
            current_hand=my_hand,
            turn=turn,
            hand_size=env.hand_size,
            round_idx=int(info["round_idx"]),
            num_rounds=int(info["num_rounds"]),
            my_pudding_count=int(info["my_pudding"]),
            opp_pudding_count=int(info["opp_pudding"]),
            my_total_score=float(info["my_score"]) if "my_breakdown" in info else float(info["total_scores"][0]),
            opp_total_score=float(info["opp_score"]) if "opp_breakdown" in info else float(info["total_scores"][1]),
            action_mask=action_mask,
            topk=topk,
        )

    def build_policy_input_summary(
        self,
        policy_input: PolicyInput,
        hand_size: int,
        topk: int = 3,
    ) -> dict[str, Any]:
        """Build summary for an arbitrary player perspective."""
        return self._summary_from_components(
            my_played=list(policy_input.my_played),
            opp_played=list(policy_input.opp_played),
            current_hand=list(policy_input.hand),
            turn=int(policy_input.turn),
            hand_size=hand_size,
            round_idx=int(policy_input.round_idx),
            num_rounds=int(policy_input.num_rounds),
            my_pudding_count=int(policy_input.my_pudding_count),
            opp_pudding_count=int(policy_input.opp_pudding_count),
            my_total_score=float(policy_input.my_total_score),
            opp_total_score=float(policy_input.opp_total_score),
            action_mask=np.asarray(policy_input.action_mask, dtype=np.bool_),
            topk=topk,
        )

    def explain_agent_move(self, state_summary: dict[str, Any], chosen_action: int, topk: int = 3) -> str:
        """Explain one selected move grounded only in provided summary JSON."""
        fallback_text = self._fallback_explain(state_summary=state_summary, chosen_action=chosen_action, topk=topk)
        cache_key = self._cache_key(
            mode="explain",
            state_summary=state_summary,
            prompt_version=self._prompt_versions["explain"],
            chosen_action=chosen_action,
        )
        text, used_fallback = self._generate_with_provider(
            mode="explain",
            system_prompt=self._prompts["explain_system"],
            user_prompt=self._render_user_prompt(
                template=self._prompts["explain_user"],
                state_summary=state_summary,
                chosen_action=chosen_action,
                topk=topk,
            ),
            fallback_text=fallback_text,
            cache_key=cache_key,
        )
        self._log_response(
            mode="explain",
            state_summary=state_summary,
            chosen_action=chosen_action,
            topk=topk,
            response_text=text,
            fallback_used=used_fallback,
        )
        return text

    def coach_user(self, state_summary: dict[str, Any], topk: int = 3) -> str:
        """Recommend top actions and tradeoffs for the current user turn."""
        recommendations = state_summary.get("agent_recommendation", [])[: max(0, topk)]
        fallback_text = self._fallback_coach(state_summary=state_summary, topk=topk)
        cache_key = self._cache_key(
            mode="coach",
            state_summary=state_summary,
            prompt_version=self._prompt_versions["coach"],
            chosen_action=None,
        )
        text, used_fallback = self._generate_with_provider(
            mode="coach",
            system_prompt=self._prompts["coach_system"],
            user_prompt=self._render_user_prompt(
                template=self._prompts["coach_user"],
                state_summary=state_summary,
                chosen_action=None,
                topk=topk,
            ),
            fallback_text=fallback_text,
            cache_key=cache_key,
        )
        text, render_fallback = self._render_coach_response(
            raw_text=text,
            recommendations=recommendations,
            state_summary=state_summary,
        )
        used_fallback = used_fallback or render_fallback
        self._log_response(
            mode="coach",
            state_summary=state_summary,
            chosen_action=None,
            topk=topk,
            response_text=text,
            fallback_used=used_fallback,
        )
        return text

    def summarize_post_game(
        self,
        state_summary: dict[str, Any],
        my_score: float,
        opp_score: float,
        my_breakdown: Any,
        opp_breakdown: Any,
    ) -> str:
        """Provide post-game recap in same logging/LLM pathway."""
        final_payload = {
            "state_summary": state_summary,
            "my_score": float(my_score),
            "opp_score": float(opp_score),
            "score_diff": float(my_score - opp_score),
            "my_breakdown": self._jsonable(my_breakdown),
            "opp_breakdown": self._jsonable(opp_breakdown),
        }
        fallback_text = (
            f"Final score: you {my_score:.1f}, opponent {opp_score:.1f} "
            f"(diff {my_score - opp_score:+.1f}). Focus on higher-probability moves from coach mode "
            "when preserving key combos (Tempura pairs, Sashimi triples, and Wasabi->Nigiri timing)."
        )
        cache_key = self._cache_key(
            mode="post_game",
            state_summary=final_payload,
            prompt_version=self._prompt_versions["post_game"],
            chosen_action=None,
        )
        text, used_fallback = self._generate_with_provider(
            mode="post_game",
            system_prompt=self._prompts["coach_system"],
            user_prompt="Post-game summary request.\nSTATE_JSON:\n" + json.dumps(final_payload, sort_keys=True),
            fallback_text=fallback_text,
            cache_key=cache_key,
        )
        self._log_response(
            mode="post_game",
            state_summary=final_payload,
            chosen_action=None,
            topk=3,
            response_text=text,
            fallback_used=used_fallback,
        )
        return text

    @staticmethod
    def _action_hand_size(action_mask: np.ndarray, fallback: int) -> int:
        action_dim = int(len(action_mask))
        inferred = int(round(action_dim ** 0.5))
        return inferred if inferred > 0 and inferred * inferred == action_dim else fallback

    def _summary_from_components(
        self,
        my_played: list[str],
        opp_played: list[str],
        current_hand: list[str],
        turn: int,
        hand_size: int,
        round_idx: int,
        num_rounds: int,
        my_pudding_count: int,
        opp_pudding_count: int,
        my_total_score: float,
        opp_total_score: float,
        action_mask: np.ndarray,
        topk: int,
    ) -> dict[str, Any]:
        obs = SushiGoEnv.encode_observation(
            my_played=my_played,
            opp_played=opp_played,
            my_hand=current_hand,
            turn=turn,
            current_hand_size=len(current_hand),
            round_idx=round_idx,
            my_pudding_count=my_pudding_count,
            opp_pudding_count=opp_pudding_count,
            my_total_score=my_total_score,
            opp_total_score=opp_total_score,
            hand_size=hand_size,
        )
        legal_indices = [int(idx) for idx in np.flatnonzero(action_mask)]
        recommendations = self._recommend_actions(
            obs=obs,
            action_mask=action_mask,
            current_hand=current_hand,
            topk=topk,
        )
        return {
            "turn": int(turn),
            "round_idx": int(round_idx),
            "num_rounds": int(num_rounds),
            "hand_size": int(hand_size),
            "cards_left": int(len(current_hand)),
            "my_played_counts": {card: int(count) for card, count in rules.count_cards(my_played).items()},
            "opp_played_counts": {card: int(count) for card, count in rules.count_cards(opp_played).items()},
            "my_maki_icons": int(rules.count_maki_icons(my_played)),
            "opp_maki_icons": int(rules.count_maki_icons(opp_played)),
            "my_unpaired_wasabi": int(rules.count_available_wasabi(my_played)),
            "my_pudding_count": int(my_pudding_count),
            "opp_pudding_count": int(opp_pudding_count),
            "my_total_score": float(my_total_score),
            "opp_total_score": float(opp_total_score),
            "current_hand": [str(card) for card in current_hand],
            "action_mask_indices": legal_indices,
            "agent_recommendation": recommendations,
        }

    def _recommend_actions(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        current_hand: Sequence[str],
        topk: int,
    ) -> list[dict[str, Any]]:
        action_hand_size = self._action_hand_size(action_mask, fallback=max(len(current_hand), rules.HAND_SIZE))
        if self.policy_advisor is None:
            legal = np.flatnonzero(action_mask).tolist()
            if not legal:
                return []
            prob = 1.0 / float(len(legal))
            recommendations: list[dict[str, Any]] = []
            for idx in legal[: max(0, topk)]:
                cards = SushiGoEnv.cards_for_action(int(idx), current_hand, hand_size=action_hand_size)
                recommendations.append(
                    {
                        "action_index": int(idx),
                        "card": cards[0] if len(cards) == 1 else "+".join(cards),
                        "action_label": SushiGoEnv.describe_action(int(idx), current_hand, hand_size=action_hand_size),
                        "probability": float(prob),
                    }
                )
            return recommendations
        return self.policy_advisor.action_recommendations(
            obs=obs,
            action_mask=action_mask,
            hand_cards=current_hand,
            topk=topk,
        )

    def _generate_with_provider(
        self,
        mode: str,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        cache_key: str | None,
    ) -> tuple[str, bool]:
        if mode == "explain":
            result = self.provider.generate_explain(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback_text=fallback_text,
                cache_key=cache_key,
            )
        else:
            result = self.provider.generate_coach(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback_text=fallback_text,
                cache_key=cache_key,
            )
        return result.text, result.fallback_used

    @staticmethod
    def _prompt_version(system_prompt: str, user_prompt: str) -> str:
        material = f"{system_prompt}\n----\n{user_prompt}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _cache_key(
        mode: str,
        state_summary: dict[str, Any],
        prompt_version: str,
        chosen_action: int | None,
    ) -> str:
        summary_text = json.dumps(state_summary, sort_keys=True)
        summary_hash = hashlib.sha256(summary_text.encode("utf-8")).hexdigest()[:20]
        action_part = "none" if chosen_action is None else str(int(chosen_action))
        return f"{mode}|{summary_hash}|{prompt_version}|{action_part}"

    def _render_user_prompt(
        self,
        template: str,
        state_summary: dict[str, Any],
        chosen_action: int | None,
        topk: int,
    ) -> str:
        rendered = template.replace("{{STATE_JSON}}", json.dumps(state_summary, sort_keys=True))
        rendered = rendered.replace("{{TOPK}}", str(int(topk)))
        rendered = rendered.replace("{{CHOSEN_ACTION}}", "null" if chosen_action is None else str(int(chosen_action)))
        return rendered

    def _fallback_explain(self, state_summary: dict[str, Any], chosen_action: int, topk: int) -> str:
        hand = state_summary.get("current_hand", [])
        hand_size = int(state_summary.get("hand_size", rules.HAND_SIZE))
        action_label = SushiGoEnv.describe_action(int(chosen_action), hand, hand_size=hand_size)
        chosen_cards = SushiGoEnv.cards_for_action(int(chosen_action), hand, hand_size=hand_size)
        chosen_card = chosen_cards[0] if len(chosen_cards) == 1 else "+".join(chosen_cards)
        top = state_summary.get("agent_recommendation", [])[: max(0, topk)]
        lead = top[0] if top else None
        lead_text = "no legal recommendation available"
        if lead is not None:
            lead_text = (
                f"top suggestion was {lead.get('action_label', lead['card'])} "
                f"with probability {lead['probability']:.3f}"
            )
        alternative = self._best_alternative(top, skip_action_index=int(chosen_action))
        if alternative is None:
            tradeoff_text = "No meaningful alternative remained among the top legal moves."
        else:
            alt_label = str(alternative.get("action_label", f"idx {alternative['action_index']} -> {alternative['card']}"))
            alt_reason = self._card_tradeoff(str(alternative["card"]), state_summary).rstrip(".")
            tradeoff_text = f"Alternative {alt_label} instead {alt_reason}."
        alignment = "matches" if lead and int(lead["action_index"]) == int(chosen_action) else "differs from"
        return (
            f"Move: {action_label}. Why: This {alignment} the highest-ranked move: {lead_text}. "
            f"It favors the line where {self._card_tradeoff(chosen_card, state_summary).rstrip('.')}. "
            f"Tradeoff: {tradeoff_text} "
            f"Round {state_summary.get('round_idx', 0) + 1}/{state_summary.get('num_rounds', 1)} has "
            f"maki race {state_summary['my_maki_icons']} vs {state_summary['opp_maki_icons']}, "
            f"unpaired wasabi {state_summary['my_unpaired_wasabi']}, "
            f"and pudding {state_summary.get('my_pudding_count', 0)} vs {state_summary.get('opp_pudding_count', 0)}."
        )

    def _fallback_coach(self, state_summary: dict[str, Any], topk: int) -> str:
        recommendations = state_summary.get("agent_recommendation", [])[: max(0, topk)]
        if not recommendations:
            return "No legal moves remain for this turn."

        return self._format_coach_response(
            state_summary=state_summary,
            recommendations=recommendations,
            details_by_action=None,
        )

    def _render_coach_response(
        self,
        raw_text: str,
        recommendations: Sequence[dict[str, Any]],
        state_summary: dict[str, Any],
    ) -> tuple[str, bool]:
        if not recommendations:
            text = raw_text.strip()
            return ("No legal moves remain for this turn.", text != "No legal moves remain for this turn.")

        payload = self._parse_coach_payload(raw_text)
        if payload is None:
            return self._fallback_coach(state_summary=state_summary, topk=len(recommendations)), True

        allowed = {int(rec["action_index"]) for rec in recommendations}
        details_by_action: dict[int, dict[str, str]] = {}
        for item in payload:
            if not isinstance(item, dict):
                return self._fallback_coach(state_summary=state_summary, topk=len(recommendations)), True
            try:
                action_index = int(item["action_index"])
            except (KeyError, TypeError, ValueError):
                return self._fallback_coach(state_summary=state_summary, topk=len(recommendations)), True
            if action_index not in allowed or action_index in details_by_action:
                return self._fallback_coach(state_summary=state_summary, topk=len(recommendations)), True
            reason = self._clean_generated_text(item.get("reason", ""))
            tradeoff = self._clean_generated_text(item.get("tradeoff", ""))
            if not reason or not tradeoff:
                return self._fallback_coach(state_summary=state_summary, topk=len(recommendations)), True
            details_by_action[action_index] = {"reason": reason, "tradeoff": tradeoff}

        if set(details_by_action) != allowed:
            return self._fallback_coach(state_summary=state_summary, topk=len(recommendations)), True

        return (
            self._format_coach_response(
                state_summary=state_summary,
                recommendations=recommendations,
                details_by_action=details_by_action,
            ),
            False,
        )

    def _format_coach_response(
        self,
        state_summary: dict[str, Any],
        recommendations: Sequence[dict[str, Any]],
        details_by_action: dict[int, dict[str, str]] | None,
    ) -> str:
        lines = ["Top moves:"]
        if state_summary.get("num_rounds", 1) > 1:
            lines.append(
                "Game state: "
                f"round {state_summary.get('round_idx', 0) + 1}/{state_summary.get('num_rounds', 1)}, "
                f"pudding {state_summary.get('my_pudding_count', 0)} vs {state_summary.get('opp_pudding_count', 0)}, "
                f"score {state_summary.get('my_total_score', 0.0):.1f} vs {state_summary.get('opp_total_score', 0.0):.1f}."
            )
        for rank, rec in enumerate(recommendations, start=1):
            action_index = int(rec["action_index"])
            action_label = str(rec.get("action_label", f"idx {rec['action_index']} -> {rec['card']}"))
            detail = None if details_by_action is None else details_by_action.get(action_index)
            if detail is None:
                reason = self._clean_generated_text(self._card_tradeoff(str(rec["card"]), state_summary))
                tradeoff = self._fallback_tradeoff(
                    action_index=action_index,
                    recommendations=recommendations,
                    state_summary=state_summary,
                )
            else:
                reason = detail["reason"]
                tradeoff = detail["tradeoff"]
            lines.append(
                f"{rank}) {action_label} (p={rec['probability']:.3f}): {self._ensure_sentence(reason)} "
                f"Tradeoff: {self._ensure_sentence(tradeoff)}"
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_json_snippet(text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        for opener, closer in (("[", "]"), ("{", "}")):
            start = stripped.find(opener)
            end = stripped.rfind(closer)
            if start != -1 and end != -1 and end > start:
                return stripped[start : end + 1]
        return None

    def _parse_coach_payload(self, text: str) -> list[dict[str, Any]] | None:
        snippet = self._extract_json_snippet(text)
        if snippet is None:
            return None
        try:
            payload = json.loads(snippet)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            payload = payload.get("recommendations")
        if not isinstance(payload, list):
            return None
        return payload

    @staticmethod
    def _clean_generated_text(value: Any) -> str:
        text = str(value).strip()
        text = re.sub(r"\s+", " ", text)
        return text.strip(" -\n\t\"'")

    @staticmethod
    def _ensure_sentence(text: str) -> str:
        cleaned = LLMAssistant._clean_generated_text(text)
        if not cleaned:
            return ""
        if cleaned[-1] in ".!?":
            return cleaned
        return cleaned + "."

    @staticmethod
    def _best_alternative(
        recommendations: Sequence[dict[str, Any]],
        skip_action_index: int,
    ) -> dict[str, Any] | None:
        for rec in recommendations:
            if int(rec["action_index"]) != int(skip_action_index):
                return rec
        return None

    def _fallback_tradeoff(
        self,
        action_index: int,
        recommendations: Sequence[dict[str, Any]],
        state_summary: dict[str, Any],
    ) -> str:
        alternative = self._best_alternative(recommendations, skip_action_index=action_index)
        if alternative is None:
            return "No alternative remains because this is the only legal move."
        alt_label = str(alternative.get("action_label", f"idx {alternative['action_index']} -> {alternative['card']}"))
        alt_reason = self._card_tradeoff(str(alternative["card"]), state_summary).rstrip(".")
        return f"Skipping {alt_label} means passing on an option that {alt_reason}."

    @staticmethod
    def _card_tradeoff(card: str, state_summary: dict[str, Any]) -> str:
        if card == rules.TEMPURA:
            return "better when you can complete/keep a tempura pair."
        if card == rules.SASHIMI:
            return "best when you can reach multiples of three sashimi."
        if card == rules.DUMPLING:
            return "increases with set size, but marginal gain drops near cap."
        if card in rules.NIGIRI_VALUES:
            if int(state_summary.get("my_unpaired_wasabi", 0)) > 0:
                return "high value now because unpaired wasabi can triple nigiri."
            return "solid immediate points, especially higher nigiri values."
        if card == rules.WASABI:
            return "setup card; strongest if you can follow with high nigiri soon."
        if card == rules.CHOPSTICKS:
            return "lets you convert a later turn into a two-card combo turn."
        if card == rules.PUDDING:
            return "scores at end of game, so it matters across rounds rather than immediately."
        if "+" in card:
            return "two-card chopsticks line that combines immediate tempo with setup value."
        if card in rules.MAKI_ICONS:
            my_maki = int(state_summary.get("my_maki_icons", 0))
            opp_maki = int(state_summary.get("opp_maki_icons", 0))
            return f"affects maki race (current {my_maki} vs {opp_maki})."
        return "balances immediate value and future combo flexibility."

    def _log_response(
        self,
        mode: str,
        state_summary: dict[str, Any],
        chosen_action: int | None,
        topk: int,
        response_text: str,
        fallback_used: bool,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "state_summary": self._jsonable(state_summary),
            "chosen_action": None if chosen_action is None else int(chosen_action),
            "topk_actions": self._jsonable(state_summary.get("agent_recommendation", [])[: max(0, topk)]),
            "response_text": str(response_text),
            "provider": self.provider_name,
            "model_name": self.model_name,
            "fallback_mode": bool(fallback_used),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _load_prompt(self, filename: str, default_text: str) -> str:
        path = self.prompts_dir / filename
        if not path.exists():
            return default_text
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _default_explain_system_prompt() -> str:
        return (
            "You are a Sushi Go move explainer. Use only STATE_JSON facts. "
            "Never invent hidden cards, deck order, or opponent intent."
        )

    @staticmethod
    def _default_explain_user_prompt() -> str:
        return (
            "STATE_JSON:\n{{STATE_JSON}}\n\n"
            "Chosen action index: {{CHOSEN_ACTION}}\n"
            "TopK: {{TOPK}}\n"
            "Explain briefly why the action is reasonable and mention one tradeoff."
        )

    @staticmethod
    def _default_coach_system_prompt() -> str:
        return (
            "You are a Sushi Go coach. Use only STATE_JSON facts. "
            "Do not claim hidden information. Be concise and actionable."
        )

    @staticmethod
    def _default_coach_user_prompt() -> str:
        return (
            "STATE_JSON:\n{{STATE_JSON}}\n\n"
            "Return JSON only. Use exactly the action_index values already present in agent_recommendation.\n"
            "Schema:\n"
            "{\n"
            '  "recommendations": [\n'
            '    {"action_index": 0, "reason": "short state-grounded reason", "tradeoff": "short comparison to another legal line"}\n'
            "  ]\n"
            "}\n"
            "Include exactly {{TOPK}} items, one per recommended action."
        )

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): LLMAssistant._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [LLMAssistant._jsonable(v) for v in value]
        if is_dataclass(value):
            return {k: LLMAssistant._jsonable(v) for k, v in asdict(value).items()}
        return str(value)


def evaluate_explanations(
    policy: PolicyAdvisor | Callable[..., int],
    env: SushiGoEnv,
    num_episodes: int,
    mode: str,
    assistant: LLMAssistant | None = None,
    topk: int = 3,
    seed: int = 123,
    csv_path: Path = Path("runs/llm_logs/llm_evaluation.csv"),
) -> dict[str, Any]:
    """Run LLM explain/coach generation over trajectories and log aggregate metrics."""
    if mode not in {"explain", "coach"}:
        raise ValueError(f"mode must be 'explain' or 'coach', got {mode!r}")
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")

    llm = assistant
    if llm is None:
        llm = LLMAssistant(policy_advisor=policy if isinstance(policy, PolicyAdvisor) else None)

    generated_texts: list[str] = []
    total_steps = 0

    for episode_idx in range(num_episodes):
        obs, info = env.reset(seed=seed + episode_idx)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            mask = env.action_masks()
            summary = llm.build_state_summary(env=env, info=info, topk=topk)
            action = _policy_action(policy=policy, env=env, obs=obs, action_mask=mask)

            if mode == "explain":
                text = llm.explain_agent_move(state_summary=summary, chosen_action=action, topk=topk)
            else:
                text = llm.coach_user(state_summary=summary, topk=topk)

            generated_texts.append(text)
            total_steps += 1
            obs, _, terminated, truncated, info = env.step(int(action))

    metrics = _compute_text_metrics(generated_texts)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "episodes": int(num_episodes),
        "steps": int(total_steps),
        "avg_words": float(metrics["avg_words"]),
        "key_term_rate": float(metrics["key_term_rate"]),
        "distinct_ratio": float(metrics["distinct_ratio"]),
        "provider": llm.provider_name,
        "model_name": llm.model_name,
        "fallback_mode": bool(llm.fallback_mode),
    }
    _append_eval_csv(csv_path, row)
    return row


def _policy_action(
    policy: PolicyAdvisor | Callable[..., int],
    env: SushiGoEnv,
    obs: np.ndarray,
    action_mask: np.ndarray,
) -> int:
    if isinstance(policy, PolicyAdvisor):
        return int(policy.predict_action(obs=obs, action_mask=action_mask, deterministic=True))

    if not callable(policy):
        raise TypeError("policy must be PolicyAdvisor or callable")

    try:
        return int(policy(obs, action_mask))
    except TypeError:
        policy_input = env._policy_input_for_player(player_index=0, mask=action_mask)
        return int(policy(policy_input))


def _compute_text_metrics(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {"avg_words": 0.0, "key_term_rate": 0.0, "distinct_ratio": 0.0}

    normalized = [text.strip().lower() for text in texts]
    word_counts = [len(text.split()) for text in normalized]
    avg_words = float(np.mean(word_counts))

    key_hits = 0
    for text in normalized:
        if any(term in text for term in STRATEGIC_TERMS):
            key_hits += 1
    key_term_rate = float(key_hits / len(normalized))
    distinct_ratio = float(len(set(normalized)) / len(normalized))
    return {
        "avg_words": avg_words,
        "key_term_rate": key_term_rate,
        "distinct_ratio": distinct_ratio,
    }


def _append_eval_csv(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "mode",
        "episodes",
        "steps",
        "avg_words",
        "key_term_rate",
        "distinct_ratio",
        "provider",
        "model_name",
        "fallback_mode",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


__all__ = [
    "DEFAULT_OPENAI_MODEL",
    "LLMAssistant",
    "PolicyAdvisor",
    "evaluate_explanations",
]
