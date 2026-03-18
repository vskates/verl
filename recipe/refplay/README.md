# RefPlay

`RefPlay` trains an actor model by directly competing against a frozen reference model on the same prompts.

Pipeline:

1. The actor generates one response for each prompt.
2. The frozen reference model generates one response for the same prompt.
3. The reward function scores both responses independently.
4. The higher-scoring response becomes `chosen`, the other becomes `rejected`.
5. The actor is updated with a DPO-style loss against frozen reference log-probs.

Implementation notes:

- Built on top of the existing `SPPO`/`SPIN` recipe layer.
- Does not update the reference model.
- Does not require editing pyc-only core `verl/*`.
- Current MVP assumes `actor_rollout_ref.rollout.n=1` and `reference_rollout.rollout.n=1`.
