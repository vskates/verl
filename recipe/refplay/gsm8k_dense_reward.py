import re
from dataclasses import dataclass

import torch


_FINAL_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize_number_str(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    text = text.replace(",", "")
    text = text.strip(". ")
    return text or None


def _extract_candidate_answer_with_meta(text: str) -> tuple[str | None, bool, str]:
    if not text:
        return None, False, "none"

    match = _FINAL_ANSWER_RE.search(text)
    if match is not None:
        answer_region = match.group(1)
        numbers = _NUMBER_RE.findall(answer_region)
        if numbers:
            return _normalize_number_str(numbers[-1]), True, "hash"
        normalized = _normalize_number_str(answer_region)
        if normalized is not None:
            return normalized, True, "hash"
        return None, True, "hash"

    return None, False, "none"


def _extract_candidate_answer(text: str) -> tuple[str | None, bool]:
    answer, has_final_marker, _ = _extract_candidate_answer_with_meta(text)
    return answer, has_final_marker


def _extract_ground_truth(reward_model_entry, extra_info_entry) -> str | None:
    if isinstance(reward_model_entry, dict):
        ground_truth = reward_model_entry.get("ground_truth")
        if ground_truth is not None:
            return _normalize_number_str(str(ground_truth))
    if isinstance(extra_info_entry, dict):
        answer = extra_info_entry.get("answer")
        if isinstance(answer, str):
            extracted, _ = _extract_candidate_answer(answer)
            if extracted is not None:
                return extracted
    return None


def _extract_last_number_fallback(text: str) -> str | None:
    if not text:
        return None
    numbers = _NUMBER_RE.findall(text)
    if not numbers:
        return None
    return _normalize_number_str(numbers[-1])


def _parse_numeric_value(text: str | None) -> float | None:
    normalized = _normalize_number_str(text)
    if normalized is None:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _compute_numeric_proximity(predicted_answer: str | None, ground_truth: str | None) -> tuple[float, float | None, float | None, float | None]:
    predicted_value = _parse_numeric_value(predicted_answer)
    ground_truth_value = _parse_numeric_value(ground_truth)
    if predicted_value is None or ground_truth_value is None:
        return 0.0, predicted_value, ground_truth_value, None

    if predicted_value == ground_truth_value:
        return 1.0, predicted_value, ground_truth_value, 0.0

    denominator = max(abs(ground_truth_value), 1.0)
    relative_error = abs(predicted_value - ground_truth_value) / denominator
    proximity = 1.0 / (1.0 + relative_error)
    return float(proximity), predicted_value, ground_truth_value, float(relative_error)


def _score_gsm8k_dense_response(
    response_text: str,
    reward_model_entry,
    extra_info_entry,
    *,
    correct_reward: float = 1.0,
    incorrect_reward: float = 0.05,
    no_answer_reward: float = 0.0,
    format_bonus: float = 0.02,
    answer_bonus: float = 0.03,
    proximity_bonus: float = 0.20,
    max_incorrect_reward: float = 0.30,
    allow_fallback_answer: bool = True,
) -> tuple[float, dict[str, float | str]]:
    ground_truth = _extract_ground_truth(reward_model_entry, extra_info_entry)
    predicted_answer, has_final_marker, final_marker_type = _extract_candidate_answer_with_meta(response_text)
    used_fallback_answer = False
    if predicted_answer is None and allow_fallback_answer:
        predicted_answer = _extract_last_number_fallback(response_text)
        used_fallback_answer = predicted_answer is not None

    has_answer = predicted_answer is not None
    exact_match = float(has_answer and ground_truth is not None and predicted_answer == ground_truth)
    numeric_proximity, predicted_value, ground_truth_value, relative_error = _compute_numeric_proximity(
        predicted_answer,
        ground_truth,
    )

    if exact_match > 0.0:
        score = float(correct_reward)
    else:
        score = float(no_answer_reward)
        if has_final_marker:
            score += float(format_bonus)
        if has_answer:
            score = max(score, float(incorrect_reward))
            score += float(answer_bonus)
            score += float(proximity_bonus) * float(numeric_proximity)
        score = min(score, float(max_incorrect_reward))

    metadata = {
        "ground_truth": ground_truth or "",
        "predicted_answer": predicted_answer or "",
        "exact_match": float(exact_match),
        "has_final_marker": float(has_final_marker),
        "final_marker_type": final_marker_type,
        "has_answer": float(has_answer),
        "used_fallback_answer": float(used_fallback_answer),
        "numeric_ground_truth": "" if ground_truth_value is None else str(ground_truth_value),
        "numeric_prediction": "" if predicted_value is None else str(predicted_value),
        "numeric_relative_error": 0.0 if relative_error is None else float(relative_error),
        "numeric_proximity": float(numeric_proximity),
        "dense_reward": float(score),
    }
    return float(score), metadata


@dataclass
class RuleBasedGSM8KReward:
    tokenizer: object
    num_examine: int = 0
    correct_reward: float = 1.0
    incorrect_reward: float = 0.1
    no_answer_reward: float = 0.0

    def _decode_rows(self, rows) -> list[str]:
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in rows]

    def __call__(self, batch, return_dict: bool = False):
        response_mask = batch.batch["response_mask"]
        reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)

        reward_model_entries = batch.non_tensor_batch.get("reward_model", [None] * response_mask.shape[0])
        extra_info_entries = batch.non_tensor_batch.get("extra_info", [None] * response_mask.shape[0])
        responses = self._decode_rows(batch.batch["responses"])

        reward_extra_info = {
            "ground_truth": [],
            "predicted_answer": [],
            "exact_match": [],
            "has_final_marker": [],
            "final_marker_type": [],
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

            valid_len = int(response_mask[idx].sum().item())
            if valid_len > 0:
                reward_tensor[idx, valid_len - 1] = score

            reward_extra_info["ground_truth"].append(ground_truth or "")
            reward_extra_info["predicted_answer"].append(predicted_answer or "")
            reward_extra_info["exact_match"].append(float(exact_match))
            reward_extra_info["has_final_marker"].append(float(has_final_marker))
            reward_extra_info["final_marker_type"].append("hash" if has_final_marker else "none")
            reward_extra_info["has_answer"].append(float(has_answer))
            reward_extra_info["rule_reward"].append(float(score))

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor


@dataclass
class DenseGSM8KReward:
    tokenizer: object
    num_examine: int = 0
    correct_reward: float = 1.0
    incorrect_reward: float = 0.05
    no_answer_reward: float = 0.0
    format_bonus: float = 0.02
    answer_bonus: float = 0.03
    proximity_bonus: float = 0.20
    max_incorrect_reward: float = 0.30
    allow_fallback_answer: bool = True

    def _decode_rows(self, rows) -> list[str]:
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in rows]

    def __call__(self, batch, return_dict: bool = False):
        response_mask = batch.batch["response_mask"]
        reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)

        reward_model_entries = batch.non_tensor_batch.get("reward_model", [None] * response_mask.shape[0])
        extra_info_entries = batch.non_tensor_batch.get("extra_info", [None] * response_mask.shape[0])
        responses = self._decode_rows(batch.batch["responses"])

        reward_extra_info = {
            "ground_truth": [],
            "predicted_answer": [],
            "exact_match": [],
            "has_final_marker": [],
            "final_marker_type": [],
            "has_answer": [],
            "used_fallback_answer": [],
            "numeric_ground_truth": [],
            "numeric_prediction": [],
            "numeric_relative_error": [],
            "numeric_proximity": [],
            "dense_reward": [],
        }

        for idx, response_text in enumerate(responses):
            reward_model_entry = reward_model_entries[idx] if idx < len(reward_model_entries) else None
            extra_info_entry = extra_info_entries[idx] if idx < len(extra_info_entries) else None
            score, metadata = _score_gsm8k_dense_response(
                response_text,
                reward_model_entry,
                extra_info_entry,
                correct_reward=self.correct_reward,
                incorrect_reward=self.incorrect_reward,
                no_answer_reward=self.no_answer_reward,
                format_bonus=self.format_bonus,
                answer_bonus=self.answer_bonus,
                proximity_bonus=self.proximity_bonus,
                max_incorrect_reward=self.max_incorrect_reward,
                allow_fallback_answer=self.allow_fallback_answer,
            )

            valid_len = int(response_mask[idx].sum().item())
            if valid_len > 0:
                reward_tensor[idx, valid_len - 1] = score

            for key in reward_extra_info:
                reward_extra_info[key].append(metadata[key])

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
