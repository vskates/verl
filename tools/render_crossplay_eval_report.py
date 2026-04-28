import argparse
import json
import math
import os
from pathlib import Path


RUN_ORDER = [
    "init_vs_init",
    "initA_vs_ckptB",
    "ckpt_vs_ckpt",
    "ckptA_vs_initB",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render local CrossPlay eval report from JSON metrics.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-step", required=True)
    return parser.parse_args()


def load_runs(data_dir: Path) -> dict:
    runs = {}
    for run_name in RUN_ORDER:
        matches = sorted(data_dir.glob(f"{run_name}_*.json"))
        if not matches:
            raise FileNotFoundError(f"Missing JSON for run {run_name!r} in {data_dir}")
        with matches[-1].open("r", encoding="utf-8") as f:
            runs[run_name] = json.load(f)
    return runs


def metric(payload: dict, name: str) -> float:
    return float(payload["metrics"][name])


def write_summary_tsv(output_dir: Path, runs: dict) -> None:
    rows = [
        "run\tpolicy_a_win_rate\tpolicy_b_win_rate\ttie_rate\tpolicy_a_reward_mean\tpolicy_b_reward_mean",
    ]
    for run_name in RUN_ORDER:
        payload = runs[run_name]
        rows.append(
            "\t".join(
                [
                    run_name,
                    f"{metric(payload, 'val/policy_a_win_rate'):.4f}",
                    f"{metric(payload, 'val/policy_b_win_rate'):.4f}",
                    f"{metric(payload, 'val/tie_rate'):.4f}",
                    f"{metric(payload, 'val/policy_a_reward_mean'):.4f}",
                    f"{metric(payload, 'val/policy_b_reward_mean'):.4f}",
                ]
            )
        )
    (output_dir / "summary.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def scenario_meta(step: str) -> dict:
    return {
        "init_vs_init": {
            "label": "A₀ vs B₀",
            "subtitle": "initial vs initial",
            "meaning": "Базовая стартовая точка до обучения",
        },
        "initA_vs_ckptB": {
            "label": f"A₀ vs B{to_subscript(step)}",
            "subtitle": "how much B improved",
            "meaning": "Насколько выросла модель B",
        },
        "ckpt_vs_ckpt": {
            "label": f"A{to_subscript(step)} vs B{to_subscript(step)}",
            "subtitle": "trained vs trained",
            "meaning": "Итоговое сравнение двух финальных весов",
        },
        "ckptA_vs_initB": {
            "label": f"A{to_subscript(step)} vs B₀",
            "subtitle": "how much A improved",
            "meaning": "Насколько выросла модель A",
        },
    }


def to_subscript(text: str) -> str:
    table = str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋")
    return str(text).translate(table)


def fmt_value(value: float) -> str:
    return f"{value:.2f}"


def svg_header(title: str, subtitle: str, y_max: float = 1.0) -> list[str]:
    grid = []
    for tick in range(6):
        y_val = y_max * tick / 5
        y = 570 - (460 * tick / 5)
        grid.append(f'    <line x1="110" y1="{y:.1f}" x2="1120" y2="{y:.1f}" stroke="#e5e7eb"/>')
        grid.append(
            f'    <text x="96" y="{y + 5:.1f}" text-anchor="end">{y_val:.1f}</text>'
        )
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720">',
        '  <rect width="1200" height="720" fill="#fffdf8"/>',
        f'  <text x="600" y="42" text-anchor="middle" font-family="Georgia, serif" font-size="30" fill="#111827">{title}</text>',
        f'  <text x="600" y="74" text-anchor="middle" font-family="Georgia, serif" font-size="18" fill="#6b7280">{subtitle}</text>',
        '',
        '  <line x1="110" y1="570" x2="1120" y2="570" stroke="#374151" stroke-width="2"/>',
        '  <line x1="110" y1="110" x2="110" y2="570" stroke="#374151" stroke-width="2"/>',
        '',
        '  <g font-family="Arial, sans-serif" font-size="14" fill="#4b5563">',
        *grid,
        '  </g>',
        '',
    ]


def render_rates_svg(output_dir: Path, runs: dict, step: str, sample_count: int) -> None:
    meta = scenario_meta(step)
    parts = svg_header(
        f"CrossPlay outcome rates on {sample_count} GSM8K validation samples",
        f"A = Qwen2.5-0.5B, B = SmolLM2-1.7B, checkpoint = {step}",
    )
    parts.extend(
        [
            '  <g font-family="Arial, sans-serif" font-size="16" fill="#111827">',
            '    <rect x="810" y="96" width="18" height="18" rx="3" fill="#0f766e"/>',
            '    <text x="838" y="111">Win rate(A)</text>',
            '    <rect x="810" y="126" width="18" height="18" rx="3" fill="#b45309"/>',
            '    <text x="838" y="141">Win rate(B)</text>',
            '    <rect x="810" y="156" width="18" height="18" rx="3" fill="#64748b"/>',
            '    <text x="838" y="171">Tie rate</text>',
            '  </g>',
            '',
        ]
    )

    centers = [240, 485, 730, 975]
    for run_name, center in zip(RUN_ORDER, centers):
        payload = runs[run_name]
        values = [
            metric(payload, "val/policy_a_win_rate"),
            metric(payload, "val/policy_b_win_rate"),
            metric(payload, "val/tie_rate"),
        ]
        xs = [center - 80, center - 24, center + 32]
        colors = ["#0f766e", "#b45309", "#64748b"]
        parts.append('  <g font-family="Arial, sans-serif" font-size="13" fill="#111827">')
        for x, value, color in zip(xs, values, colors):
            height = 460 * value
            y = 570 - height
            parts.append(f'    <rect x="{x}" y="{y:.1f}" width="48" height="{height:.1f}" rx="4" fill="{color}"/>')
            if value > 0:
                parts.append(f'    <text x="{x + 24}" y="{max(100, y - 10):.1f}" text-anchor="middle">{fmt_value(value)}</text>')
            else:
                parts.append(f'    <text x="{x + 24}" y="558" text-anchor="middle" fill="#6b7280">0.00</text>')
        parts.append(f'    <text x="{center}" y="603" text-anchor="middle" font-size="20">{meta[run_name]["label"]}</text>')
        parts.append(f'    <text x="{center}" y="628" text-anchor="middle" fill="#6b7280">{meta[run_name]["subtitle"]}</text>')
        parts.append("  </g>")
        parts.append("")

    parts.append("</svg>")
    (output_dir / "crossplay_eval_rates.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_rewards_svg(output_dir: Path, runs: dict, step: str, sample_count: int) -> None:
    meta = scenario_meta(step)
    all_values = []
    for run_name in RUN_ORDER:
        all_values.extend(
            [
                metric(runs[run_name], "val/policy_a_reward_mean"),
                metric(runs[run_name], "val/policy_b_reward_mean"),
            ]
        )
    y_max = max(0.1, max(all_values))
    y_max = math.ceil(y_max * 10) / 10
    parts = svg_header(
        f"CrossPlay reward means on {sample_count} GSM8K validation samples",
        f"A = Qwen2.5-0.5B, B = SmolLM2-1.7B, checkpoint = {step}",
        y_max=y_max,
    )
    parts.extend(
        [
            '  <g font-family="Arial, sans-serif" font-size="16" fill="#111827">',
            '    <rect x="840" y="96" width="18" height="18" rx="3" fill="#0f766e"/>',
            '    <text x="868" y="111">E[r_A]</text>',
            '    <rect x="840" y="126" width="18" height="18" rx="3" fill="#b45309"/>',
            '    <text x="868" y="141">E[r_B]</text>',
            '  </g>',
            '',
        ]
    )

    centers = [240, 485, 730, 975]
    for run_name, center in zip(RUN_ORDER, centers):
        payload = runs[run_name]
        values = [
            metric(payload, "val/policy_a_reward_mean"),
            metric(payload, "val/policy_b_reward_mean"),
        ]
        xs = [center - 52, center + 12]
        colors = ["#0f766e", "#b45309"]
        parts.append('  <g font-family="Arial, sans-serif" font-size="13" fill="#111827">')
        for x, value, color in zip(xs, values, colors):
            height = 460 * (value / y_max)
            y = 570 - height
            parts.append(f'    <rect x="{x}" y="{y:.1f}" width="52" height="{height:.1f}" rx="4" fill="{color}"/>')
            if value > 0:
                parts.append(f'    <text x="{x + 26}" y="{max(100, y - 10):.1f}" text-anchor="middle">{value:.3f}</text>')
            else:
                parts.append(f'    <text x="{x + 26}" y="558" text-anchor="middle" fill="#6b7280">0.000</text>')
        parts.append(f'    <text x="{center}" y="603" text-anchor="middle" font-size="20">{meta[run_name]["label"]}</text>')
        parts.append(f'    <text x="{center}" y="628" text-anchor="middle" fill="#6b7280">{meta[run_name]["subtitle"]}</text>')
        parts.append("  </g>")
        parts.append("")

    parts.append("</svg>")
    (output_dir / "crossplay_eval_rewards.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_html(output_dir: Path, runs: dict, step: str) -> None:
    meta = scenario_meta(step)
    samples = runs["init_vs_init"]["max_samples"]
    rows = []
    for run_name in RUN_ORDER:
        payload = runs[run_name]
        rows.append(
            f"""
              <tr>
                <td><b>{meta[run_name]['label']}</b></td>
                <td>{meta[run_name]['meaning']}</td>
                <td>{metric(payload, 'val/policy_a_win_rate'):.2f}</td>
                <td>{metric(payload, 'val/policy_b_win_rate'):.2f}</td>
                <td>{metric(payload, 'val/tie_rate'):.2f}</td>
                <td>{metric(payload, 'val/policy_a_reward_mean'):.3f}</td>
                <td>{metric(payload, 'val/policy_b_reward_mean'):.3f}</td>
              </tr>
            """.rstrip()
        )

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CrossPlay Eval Report</title>
  <style>
    :root {{
      --bg: #f7f4ec;
      --panel: #fffdf8;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #ddd6c8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #fff7ed 0, var(--bg) 42%, #efe9dd 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.45;
    }}
    .page {{ max-width: 1200px; margin: 0 auto; padding: 32px 20px 56px; }}
    .hero, .section {{
      background: rgba(255,255,255,.9);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px;
      box-shadow: 0 18px 40px rgba(60, 40, 10, .07);
    }}
    .section {{ margin-top: 22px; }}
    h1, h2 {{ margin: 0 0 10px; line-height: 1.15; }}
    h1 {{ font-size: 38px; }}
    h2 {{ font-size: 28px; margin-top: 8px; }}
    .lead {{ font-size: 19px; max-width: 920px; }}
    .muted {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }}
    .kicker {{
      font-size: 12px;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .value {{ font-size: 30px; font-weight: 700; margin-top: 6px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      background: var(--panel);
      border-radius: 14px;
      overflow: hidden;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid #ece7db;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f6efe3;
      font-size: 13px;
      letter-spacing: .03em;
      text-transform: uppercase;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .chart {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      margin-top: 14px;
    }}
    img {{ width: 100%; height: auto; display: block; }}
    code {{ background: #f3ede2; padding: 2px 6px; border-radius: 6px; }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 32px; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>CrossPlay: финальный срез обучения</h1>
      <p class="lead">
        Отчёт по четырём сценариям сравнения для пары моделей
        A = Qwen2.5-0.5B-Instruct и B = SmolLM2-1.7B-Instruct
        на задачах GSM8K после завершённого run до checkpoint {step}.
      </p>
      <p class="muted">
        Здесь мы сравниваем стартовые веса (A0, B0) и финальные веса после обучения
        (A{step}, B{step}) на первых {samples} примерах из validation split.
      </p>
      <div class="grid">
        <div class="card">
          <div class="kicker">Модели</div>
          <div class="value">A vs B</div>
          <div>A: Qwen 0.5B, B: SmolLM 1.7B</div>
        </div>
        <div class="card">
          <div class="kicker">Датасет</div>
          <div class="value">GSM8K</div>
          <div>Rule-based judge, math word problems</div>
        </div>
        <div class="card">
          <div class="kicker">Checkpoint</div>
          <div class="value">{step}</div>
          <div>Финальные веса после успешного full run</div>
        </div>
        <div class="card">
          <div class="kicker">Eval</div>
          <div class="value">{samples}</div>
          <div>Валидационных примеров в каждом сравнении</div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Сценарий обучения</h2>
      <p>
        В этом CrossPlay обе модели отвечают на одни и те же задачи GSM8K, после чего rule-based judge
        сравнивает ответы. Затем каждая сторона обновляется только на своих собственных проигрышах через replay + DPO.
      </p>
      <p>
        Поэтому четыре сравнения ниже полезны по-разному: A0 vs B0 показывает старт, A0 vs B{step} показывает,
        насколько выросла B, A{step} vs B0 показывает рост A, а A{step} vs B{step} даёт итоговую дуэль финальных весов.
      </p>
    </section>

    <section class="section">
      <h2>Результаты</h2>
      <table>
        <thead>
          <tr>
            <th>Сценарий</th>
            <th>Смысл</th>
            <th>Win rate(A)</th>
            <th>Win rate(B)</th>
            <th>Tie rate</th>
            <th>E[r_A]</th>
            <th>E[r_B]</th>
          </tr>
        </thead>
        <tbody>
{''.join(rows)}
        </tbody>
      </table>
    </section>

    <section class="section">
      <h2>Графики</h2>
      <div class="chart">
        <img src="crossplay_eval_rates.svg" alt="CrossPlay outcome rates">
      </div>
      <div class="chart">
        <img src="crossplay_eval_rewards.svg" alt="CrossPlay reward means">
      </div>
    </section>

    <section class="section">
      <h2>Что получилось на практике</h2>
      <p>
        Стартово более сильной стороной была B, но финальный checkpoint показывает заметный рост обеих моделей.
        Главный вопрос в этом отчёте — кто вырос сильнее к финалу: A или B.
      </p>
      <p class="muted">
        Сырые JSON лежат в <code>outputs/crossplay_eval_plots/data</code>, а сводка — в <code>summary.tsv</code>.
      </p>
    </section>
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(data_dir)
    sample_count = int(runs["init_vs_init"]["max_samples"])
    write_summary_tsv(output_dir, runs)
    render_rates_svg(output_dir, runs, args.checkpoint_step, sample_count)
    render_rewards_svg(output_dir, runs, args.checkpoint_step, sample_count)
    render_html(output_dir, runs, args.checkpoint_step)


if __name__ == "__main__":
    main()
