# Heterogeneous CrossPlay for Multi-Agent RL

<table>
<tr>
<td bgcolor="#f6f8fa">
<b>Contribution.</b> Heterogeneous online cross-play is studied as a multi-agent RL setting for LLM alignment, with the central hypothesis that agents learn more effectively in a dynamic opponent environment than under static self-play. A controlled cross-play setting is considered in which two live models compete on the same reasoning task, preference pairs are induced by a verifier, and both policies are optimized with matched training exposure. On GSM8K, the resulting learning dynamics are non-degenerate, competitive, and strictly stronger than initialization.<br><br>
<b>Author:</b> Ekaterina Vasiagina<br>
<b>Poster:</b> <a href="poster/poster_crossplay_a1.pdf">Poster</a>
</td>
</tr>
</table>

## Method

Two models, `Qwen2.5-0.5B-Instruct` and `SmolLM2-1.7B-Instruct`, are trained in a shared online game. Given a prompt `x`, both policies generate responses, a dense GSM8K verifier converts them into a preferred/rejected pair, and each policy is updated against its own frozen anchor with a DPO-style objective.

The resulting optimization can be written as:

```math
y_A \sim \pi_A(\cdot \mid x), \qquad
y_B \sim \pi_B(\cdot \mid x)
```

```math
(y^+, y^-) = C(x, y_A, y_B)
```

```math
z_i = \beta \left[
\log \frac{\pi_i(y^+ \mid x)}{\pi_i(y^- \mid x)}
- \log \frac{\rho_i(y^+ \mid x)}{\rho_i(y^- \mid x)}
\right]
```

```math
\mathcal{L}_i = -\log \sigma(z_i), \qquad i \in \{A, B\}
```

## Main Result

At checkpoint `1000`, Qwen win rate increases from `0.008` at initialization to `0.340` under CrossPlay, while the interaction shifts from a degenerate high-tie regime to a competitive one.

## Repository

```text
recipe/crossplay/   training pipeline
recipe/refplay/     minimal helper modules used by crossplay
recipe/spin/        minimal trainer utilities used by crossplay
tools/              evaluation and report scripts
poster/             poster source and compiled PDF
verl/               core framework code
```
