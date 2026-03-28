from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from recipe.refplay.gsm8k_dense_reward import _extract_candidate_answer, _extract_ground_truth


def _normalize_text_reward_result(result: Any, batch_size: int) -> tuple[list[float], dict[str, list[Any]]]:
    if isinstance(result, dict):
        sequence_rewards = result.get("sequence_rewards")
        if sequence_rewards is None:
            sequence_rewards = result.get("sequence_reward")
        if sequence_rewards is None:
            raise KeyError("Text judge must return sequence_reward(s) when returning a dict")
        scores = [float(value) for value in sequence_rewards]
        extras = {
            key: list(value)
            for key, value in result.get("reward_extra_info", {}).items()
        }
    else:
        scores = [float(value) for value in result]
        extras = {}

    if len(scores) != batch_size:
        raise ValueError(f"Expected {batch_size} scores from text judge, got {len(scores)}")
    return scores, extras


@dataclass
class RuleBasedGSM8KTextJudge:
    correct_reward: float = 1.0
    incorrect_reward: float = 0.1
    no_answer_reward: float = 0.0

    def score_texts(
        self,
        *,
        prompts: list[Any] | None = None,
        responses: list[str],
        benchmark_ids: list[str] | None = None,
        reward_model_entries: list[Any] | None = None,
        extra_info_entries: list[Any] | None = None,
    ) -> dict[str, list[Any]]:
        reward_model_entries = reward_model_entries or [None] * len(responses)
        extra_info_entries = extra_info_entries or [None] * len(responses)

        scores: list[float] = []
        reward_extra_info = {
            "ground_truth": [],
            "predicted_answer": [],
            "exact_match": [],
            "has_final_marker": [],
            "has_answer": [],
            "rule_reward": [],
        }

        for idx, response_text in enumerate(responses):
            reward_model_entry = reward_model_entries[idx] if idx < len(reward_model_entries) else None
            extra_info_entry = extra_info_entries[idx] if idx < len(extra_info_entries) else None
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

            scores.append(float(score))
            reward_extra_info["ground_truth"].append(ground_truth or "")
            reward_extra_info["predicted_answer"].append(predicted_answer or "")
            reward_extra_info["exact_match"].append(float(exact_match))
            reward_extra_info["has_final_marker"].append(float(has_final_marker))
            reward_extra_info["has_answer"].append(float(has_answer))
            reward_extra_info["rule_reward"].append(float(score))

        return {"sequence_rewards": scores, "reward_extra_info": reward_extra_info}


class CallableTextJudge:
    """Adapter for custom text-level judge callables."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn

    def score_texts(
        self,
        *,
        prompts: list[Any],
        responses: list[str],
        benchmark_ids: list[str],
        reward_model_entries: list[Any] | None = None,
        extra_info_entries: list[Any] | None = None,
    ) -> dict[str, list[Any]]:
        result = self.fn(
            prompts=prompts,
            responses=responses,
            benchmark_ids=benchmark_ids,
            reward_model_entries=reward_model_entries,
            extra_info_entries=extra_info_entries,
        )
        scores, extras = _normalize_text_reward_result(result, batch_size=len(responses))
        return {"sequence_rewards": scores, "reward_extra_info": extras}
