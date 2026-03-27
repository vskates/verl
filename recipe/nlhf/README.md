# NLHF

`NLHF` is a dual-policy recipe built on top of the hardened `refplay` worker
stack.

Pipeline:

1. `policy_a` generates one response for each prompt.
2. `policy_b` generates one response for the same prompts.
3. The reward function / external judge scores both full responses.
4. The higher-scoring response becomes `chosen`, the other becomes `rejected`.
5. A frozen anchor model computes reference log-probs.
6. One or both trainable policies are updated with a DPO-style pairwise loss.

Implementation notes:

- Both `policy_a` and `policy_b` are trainable actor+rollout workers.
- The anchor model is frozen and only used for reference log-probs.
- `algorithm.update_mode` controls whether updates are alternating or applied
  to both policies each step.
- Reward callables may return either `reward_tensor` or sequence-level
  `sequence_reward(s)`.
