# CrossPlay

`CrossPlay` is a benchmark-aware competitive recipe for heterogeneous models.

Pipeline:

1. `policy_a` and `policy_b` can use different model families and tokenizers.
2. Both policies generate full responses for the same prompt batch.
3. A text-level judge scores the two responses independently.
4. Each policy is updated only on examples where it lost to the opponent.
5. Updates are retokenized with the target policy tokenizer, so `Llama vs Qwen`
   style matchups are supported.
6. Each policy can optionally use its own frozen anchor (`anchor_a`,
   `anchor_b`) for DPO-style reference log-probs.

This differs from `recipe/nlhf` in two important ways:

- it is safe for cross-vendor models with different tokenizers;
- it trains each side on its own failure cases instead of applying one shared
  preference batch to both models.
- anchors are optional, so a pure duel mode can be used for lightweight smoke
  testing or reference-free training.
