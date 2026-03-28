from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from recipe.refplay.gsm8k_dense_reward import _extract_candidate_answer, _extract_ground_truth


def _normalize_pairwise_result(result: Any, batch_size: int) -> tuple[list[float], dict[str, list[Any]]]:
    if isinstance(result, dict):
        preference_probs = result.get("preference_probs")
        if preference_probs is None:
            raise KeyError("Pairwise judge must return preference_probs when returning a dict")
        probs = [float(value) for value in preference_probs]
        extras = {
            key: list(value)
            for key, value in result.get("pairwise_extra_info", {}).items()
        }
    else:
        probs = [float(value) for value in result]
        extras = {}

    if len(probs) != batch_size:
        raise ValueError(f"Expected {batch_size} pairwise probabilities, got {len(probs)}")
    return probs, extras


@dataclass
class RuleBasedGSM8KPairwiseJudge:
    correct_reward: float = 1.0
    incorrect_reward: float = 0.1
    no_answer_reward: float = 0.0
    mode: str = "pairwise_probability"
    reward_temperature: float = 1.0

    def _score_single(self, response_text: str, reward_model_entry: Any, extra_info_entry: Any) -> tuple[float, dict[str, Any]]:
        ground_truth = _extract_ground_truth(reward_model_entry, extra_info_entry)
        predicted_answer, has_final_marker = _extract_candidate_answer(response_text)
        has_answer = predicted_answer is not None
        exact_match = float(has_answer and ground_truth is not None and predicted_answer == ground_truth)

        if exact_match > 0.0:
            score = self.correct_reward
        elif has_answer:
            score = self.incorrect_reward
        else:
            score = self.no_answer_reward

        return float(score), {
            "ground_truth": ground_truth or "",
            "predicted_answer": predicted_answer or "",
            "exact_match": float(exact_match),
            "has_final_marker": float(has_final_marker),
            "has_answer": float(has_answer),
            "rule_reward": float(score),
        }

    def compare_texts(
        self,
        *,
        prompts: list[Any] | None = None,
        left_responses: list[str],
        right_responses: list[str],
        benchmark_ids: list[str] | None = None,
        reward_model_entries: list[Any] | None = None,
        extra_info_entries: list[Any] | None = None,
    ) -> dict[str, list[Any]]:
        reward_model_entries = reward_model_entries or [None] * len(left_responses)
        extra_info_entries = extra_info_entries or [None] * len(left_responses)

        preference_probs: list[float] = []
        pairwise_extra_info = {
            "left_score": [],
            "right_score": [],
            "left_exact_match": [],
            "right_exact_match": [],
            "left_win": [],
            "right_win": [],
            "tie": [],
        }

        for idx, (left_text, right_text) in enumerate(zip(left_responses, right_responses, strict=True)):
            reward_model_entry = reward_model_entries[idx] if idx < len(reward_model_entries) else None
            extra_info_entry = extra_info_entries[idx] if idx < len(extra_info_entries) else None

            left_score, left_meta = self._score_single(left_text, reward_model_entry, extra_info_entry)
            right_score, right_meta = self._score_single(right_text, reward_model_entry, extra_info_entry)

            if self.mode == "pairwise_probability":
                if left_score > right_score:
                    preference = 1.0
                elif left_score < right_score:
                    preference = 0.0
                else:
                    preference = 0.5
            elif self.mode == "bt_reward":
                temperature = max(float(self.reward_temperature), 1e-6)
                preference = 1.0 / (1.0 + math.exp(-(left_score - right_score) / temperature))
            else:
                raise ValueError(f"Unsupported pairwise judge mode: {self.mode}")

            preference_probs.append(float(preference))
            pairwise_extra_info["left_score"].append(float(left_score))
            pairwise_extra_info["right_score"].append(float(right_score))
            pairwise_extra_info["left_exact_match"].append(left_meta["exact_match"])
            pairwise_extra_info["right_exact_match"].append(right_meta["exact_match"])
            pairwise_extra_info["left_win"].append(float(left_score > right_score))
            pairwise_extra_info["right_win"].append(float(right_score > left_score))
            pairwise_extra_info["tie"].append(float(left_score == right_score))

        return {
            "preference_probs": preference_probs,
            "pairwise_extra_info": pairwise_extra_info,
        }


class CallablePairwiseJudge:
    """Adapter for custom pairwise preference callables."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn

    def compare_texts(
        self,
        *,
        prompts: list[Any],
        left_responses: list[str],
        right_responses: list[str],
        benchmark_ids: list[str],
        reward_model_entries: list[Any] | None = None,
        extra_info_entries: list[Any] | None = None,
    ) -> dict[str, list[Any]]:
        result = self.fn(
            prompts=prompts,
            left_responses=left_responses,
            right_responses=right_responses,
            benchmark_ids=benchmark_ids,
            reward_model_entries=reward_model_entries,
            extra_info_entries=extra_info_entries,
        )
        probs, extras = _normalize_pairwise_result(result, batch_size=len(left_responses))
        return {"preference_probs": probs, "pairwise_extra_info": extras}
