import re
from dataclasses import dataclass

import torch

from recipe.r1.tasks.math import compute_score as math_verify_score


_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_REPEAT_RE = re.compile(r"\b(\w+(?:\s+\w+){2,6})\b(?:[\s,.;:]+\1\b)+", re.IGNORECASE)


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

    has_final_marker = "####" in text
    search_regions = []
    if has_final_marker:
        search_regions.append(text.rsplit("####", 1)[-1])
    search_regions.append(text)

    for region in search_regions:
        boxed = _BOXED_RE.findall(region)
        if boxed:
            return _normalize_number_str(boxed[-1]), has_final_marker
        numbers = _NUMBER_RE.findall(region)
        if numbers:
            return _normalize_number_str(numbers[-1]), has_final_marker
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


def _repetition_penalty(text: str) -> float:
    if not text:
        return 0.0
    lowered = text.lower()
    if len(lowered.split()) < 24:
        return 0.0
    if _REPEAT_RE.search(lowered):
        return 0.2
    return 0.0


@dataclass
class DenseGSM8KReward:
    tokenizer: object
    num_examine: int = 0
    exact_match_reward: float = 1.0
    final_marker_bonus: float = 0.2
    parseable_answer_bonus: float = 0.05
    brevity_bonus: float = 0.02
    repetition_penalty_scale: float = 1.0

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
            "parseable_answer": [],
            "repetition_penalty": [],
            "dense_reward": [],
        }

        for idx, response_text in enumerate(responses):
            reward_model_entry = reward_model_entries[idx] if idx < len(reward_model_entries) else None
            extra_info_entry = extra_info_entries[idx] if idx < len(extra_info_entries) else None
            ground_truth = _extract_ground_truth(reward_model_entry, extra_info_entry)
            predicted_answer, has_final_marker = _extract_candidate_answer(response_text)
            parseable_answer = predicted_answer is not None

            exact_match = 0.0
            if ground_truth is not None:
                try:
                    exact_match = float(math_verify_score(response_text, ground_truth))
                except Exception:
                    exact_match = 0.0

            score = self.exact_match_reward * exact_match
            if exact_match <= 0.0:
                if has_final_marker and parseable_answer:
                    score += self.final_marker_bonus
                elif parseable_answer:
                    score += self.parseable_answer_bonus
                if 0 < len(response_text.split()) <= 96:
                    score += self.brevity_bonus

            repeat_penalty = self.repetition_penalty_scale * _repetition_penalty(response_text)
            score = max(0.0, min(self.exact_match_reward, score - repeat_penalty))

            valid_len = int(response_mask[idx].sum().item())
            if valid_len > 0:
                reward_tensor[idx, valid_len - 1] = score

            reward_extra_info["ground_truth"].append(ground_truth or "")
            reward_extra_info["predicted_answer"].append(predicted_answer or "")
            reward_extra_info["exact_match"].append(float(exact_match > 0.0))
            reward_extra_info["has_final_marker"].append(float(has_final_marker))
            reward_extra_info["parseable_answer"].append(float(parseable_answer))
            reward_extra_info["repetition_penalty"].append(float(repeat_penalty))
            reward_extra_info["dense_reward"].append(float(score))

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
