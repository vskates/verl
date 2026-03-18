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


def _extract_candidate_answer(text: str) -> tuple[str | None, bool]:
    if not text:
        return None, False

    match = _FINAL_ANSWER_RE.search(text)
    has_final_marker = match is not None
    if not has_final_marker:
        return None, False

    answer_region = match.group(1)
    numbers = _NUMBER_RE.findall(answer_region)
    if numbers:
        return _normalize_number_str(numbers[-1]), True
    normalized = _normalize_number_str(answer_region)
    if normalized is not None:
        return normalized, True
    return None, has_final_marker


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
            reward_extra_info["has_answer"].append(float(has_answer))
            reward_extra_info["rule_reward"].append(float(score))

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor


# Backward-compatible alias for the previous class name used in this recipe.
DenseGSM8KReward = RuleBasedGSM8KReward
