# verl crossplay

This `main` branch is a slim repository for the `crossplay` pipeline only.

It keeps:
- the core `verl/` framework code required by the pipeline
- `recipe/crossplay/`
- the minimal helper code from `recipe/refplay/` and `recipe/spin/` that `crossplay` imports at runtime
- `tools/` scripts used for `crossplay` evaluation and report rendering
- `poster/` with the current poster source and compiled PDF

The full research workspace, auxiliary recipes, notes, reports, and extra experiment artifacts are stored in the `dev` branch.
