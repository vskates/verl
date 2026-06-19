# CrossPlay

`CrossPlay` is a benchmark-aware competitive recipe for heterogeneous models.

## Pipeline

1. `policy_a` and `policy_b` may come from different model families and use different tokenizers.
2. Both policies generate responses for the same prompt batch.
3. A text-level judge scores both responses.
4. The duel is converted into DPO pairs and stored in per-policy replay buffers.
5. Updates are retokenized with the target policy tokenizer, so `Llama vs Qwen` style matchups are supported.
6. Each policy uses its own frozen anchor (`anchor_a`, `anchor_b`) and is optimized with reference DPO against that anchor.

## Update Modes

`algorithm.focus_failed_examples_only=true`

- Each policy trains only on examples where it lost.
- This is the original failure-focused mode.
- It is useful when the goal is targeted remediation.

`algorithm.focus_failed_examples_only=false`

- Each policy trains on all decisive pairs from the fresh batch.
- If `policy_a` wins, then both updates use `chosen=response_a`, `rejected=response_b`.
- If `policy_b` wins, then both updates use `chosen=response_b`, `rejected=response_a`.
- Ties are skipped for the current hard-DPO update.

The recommended default for fairer head-to-head training is `false`, because it keeps the number of fresh DPO pairs closer across the two policies and reduces the risk that one side improves simply because it sees more updates.

## Reward Modes

### `crossplay_gsm8k_rule`

This is the original strict reward:

- `1.0` for exact match
- `0.1` for a parseable but incorrect final answer
- `0.0` for no extractable final answer

This reward is easy to debug but too coarse for early-stage competitive training: many qualitatively different failures collapse to the same score.

### `crossplay_gsm8k_dense`

This is the new recommended reward for `GSM8K`.

It keeps exact correctness as the main signal, but makes non-exact outcomes more expressive:

- exact answer: `1.0`
- missing answer: near `0.0`
- malformed but answer-like output: small positive score
- wrong but parseable answer: base wrong-answer score
- numerically closer wrong answers: higher score than far-away wrong answers

The default dense score is:

```text
r = 1.0,                                  if exact match
r = min(
      no_answer_reward
      + 1[has_final_marker] * format_bonus
      + 1[has_answer] * (incorrect_reward + answer_bonus)
      + proximity_bonus * numeric_proximity,
      max_incorrect_reward
    ),                                    otherwise
```

where:

- `numeric_proximity = 1 / (1 + relative_error)`
- `relative_error = |pred - gt| / max(|gt|, 1)`

Default parameters:

- `correct_reward = 1.0`
- `incorrect_reward = 0.05`
- `no_answer_reward = 0.0`
- `format_bonus = 0.02`
- `answer_bonus = 0.03`
- `proximity_bonus = 0.20`
- `max_incorrect_reward = 0.30`
- `allow_fallback_answer = true`

`allow_fallback_answer=true` means that if the strict `#### ...` marker is missing, the judge can still try to extract the last numeric answer from the response. This makes the reward less brittle while still giving explicit format compliance only a small bonus. The only canonical final-answer format is:

- `#### 72`

## Why Dense Reward Is The Right First Fix

The current bottleneck in `CrossPlay` is not the training loop anymore, but the low-entropy reward signal:

- exact outcomes dominate everything
- all wrong-but-parseable answers look nearly identical
- formatting errors can overshadow reasoning quality

For `GSM8K`, a dense outcome-level reward is the lowest-risk next step:

- it keeps exact-match correctness as the anchor objective
- it reduces ties among incorrect answers
- it gives replay sampling and DPO pair construction a more informative margin
- it does not require a separate verifier model yet

This is still not a full process reward. It is a pragmatic stage-1 fix.

## Research Plan

The recommended research sequence is:

1. Keep `GSM8K` for infrastructure debugging, but replace the coarse rule reward with `crossplay_gsm8k_dense`.
2. Evaluate not only on `GSM8K`, but also on contamination-robust arithmetic benchmarks such as `GSM1K`, because public `GSM8K` performance can be inflated by memorization.
3. If the dense reward stabilizes training, move training to a richer math corpus such as `MetaMathQA` or `MATH`.
4. After that, replace the hand-crafted dense reward with a verifier-style reward or pairwise preference model.

## Recommended Datasets

### Keep now: `GSM8K`

Use it for:

- debugging the competitive training loop
- comparing `rule` vs `dense` rewards
- short controlled ablations

### Add next for evaluation: `GSM1K`

Use it for:

- checking whether gains on `GSM8K` reflect reasoning rather than contamination or benchmark memorization

### Add next for training: `MetaMathQA`

Use it for:

- larger-scale math instruction tuning
- richer reasoning supervision than plain grade-school arithmetic

### Add later for harder evaluation/training: `MATH`

Use it for:

- more difficult multi-step mathematical reasoning
- stress-testing whether `CrossPlay` learns beyond shallow answer formatting

## Literature Behind The Plan

- `Solving math word problems with process- and outcome-based feedback`
  `https://arxiv.org/abs/2211.14275`
- `Let's Verify Step by Step`
  `https://arxiv.org/abs/2305.20050`
- `Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations`
  `https://arxiv.org/abs/2312.08935`
- `A Careful Examination of Large Language Model Performance on Grade School Arithmetic`
  `https://arxiv.org/abs/2405.00332`
- `MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models`
  `https://arxiv.org/abs/2309.12284`
- `Measuring Mathematical Problem Solving With the MATH Dataset`
  `https://arxiv.org/abs/2103.03874`

## Current Recommendation

For the current codebase, the most reasonable default setup is:

- `reward.reward_manager.name=crossplay_gsm8k_dense`
- `algorithm.focus_failed_examples_only=false`
- `data.max_response_length=128`
- `data.gsm8k_answer_instruction="Please reason step by step, and put the final numeric answer on the last line as: #### <answer>"`

This keeps the game symmetric enough for early experiments, makes the reward more informative, and leaves a clean path toward verifier-based rewards later.
