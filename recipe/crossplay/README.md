# CrossPlay

`CrossPlay` is a benchmark-aware competitive recipe for heterogeneous models.

Pipeline:

1. `policy_a` and `policy_b` can use different model families and tokenizers.
2. Both policies generate full responses for the same prompt batch.
3. A text-level judge scores the two responses independently.
4. Each policy records only its own failures, grouped by benchmark.
5. Updates are sampled from this benchmark-aware failure replay, so each model
   focuses on the benchmark buckets where it systematically loses.
6. Updates are retokenized with the target policy tokenizer, so `Llama vs Qwen`
   style matchups are supported.
7. Each policy uses its own frozen anchor (`anchor_a`, `anchor_b`) and is
   optimized with reference DPO against that anchor.

This differs from `recipe/nlhf` in two important ways:

- it is safe for cross-vendor models with different tokenizers;
- it trains each side on its own failure cases instead of applying one shared
  preference batch to both models.
- each side has its own benchmark-focused replay buffer and its own frozen
  reference anchor.
