import argparse
import json
from pathlib import Path


COLORS = {
    "a": "#0f766e",
    "b": "#b45309",
    "tie": "#64748b",
    "grid": "#e5e7eb",
    "axis": "#374151",
    "text": "#111827",
    "muted": "#6b7280",
    "bg": "#fffdf8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render start-vs-checkpoint CrossPlay comparison graphs.")
    parser.add_argument("--init-json", required=True)
    parser.add_argument("--ckpt-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-label", default="Checkpoint 1000")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric(payload: dict, name: str) -> float:
    return float(payload["metrics"][name])


def svg_header(title: str, subtitle: str, y_max: float) -> list[str]:
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="640" viewBox="0 0 1080 640">',
        f'  <rect width="1080" height="640" fill="{COLORS["bg"]}"/>',
        f'  <text x="540" y="40" text-anchor="middle" font-family="Georgia, serif" font-size="28" fill="{COLORS["text"]}">{title}</text>',
        f'  <text x="540" y="70" text-anchor="middle" font-family="Georgia, serif" font-size="16" fill="{COLORS["muted"]}">{subtitle}</text>',
        f'  <line x1="95" y1="525" x2="1010" y2="525" stroke="{COLORS["axis"]}" stroke-width="2"/>',
        f'  <line x1="95" y1="110" x2="95" y2="525" stroke="{COLORS["axis"]}" stroke-width="2"/>',
        f'  <g font-family="Arial, sans-serif" font-size="13" fill="{COLORS["muted"]}">',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = 525 - 415 * tick / 5
        out.append(f'    <line x1="95" y1="{y:.1f}" x2="1010" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')
        out.append(f'    <text x="82" y="{y + 5:.1f}" text-anchor="end">{value:.2f}</text>')
    out.extend(["  </g>", ""])
    return out


def render_grouped_bars(output_path: Path, title: str, subtitle: str, labels: list[str], data: list[dict], series: list[tuple[str, str, str]], y_max: float) -> None:
    parts = svg_header(title, subtitle, y_max)
    parts.append(f'  <g font-family="Arial, sans-serif" font-size="15" fill="{COLORS["text"]}">')
    legend_x = 745
    for idx, (label, _, color) in enumerate(series):
        y = 96 + 28 * idx
        parts.append(f'    <rect x="{legend_x}" y="{y}" width="16" height="16" rx="3" fill="{color}"/>')
        parts.append(f'    <text x="{legend_x + 26}" y="{y + 13}">{label}</text>')
    parts.extend(["  </g>", ""])

    plot_left = 170
    plot_width = 760
    group_width = plot_width / len(data)
    bar_width = min(46, group_width / (len(series) + 1.6))
    gap = bar_width * 0.35
    parts.append(f'  <g font-family="Arial, sans-serif" font-size="13" fill="{COLORS["text"]}">')
    for idx, item in enumerate(data):
        center = plot_left + group_width * (idx + 0.5)
        total_width = len(series) * bar_width + (len(series) - 1) * gap
        start_x = center - total_width / 2
        for sidx, (_, key, color) in enumerate(series):
            value = item[key]
            height = 415 * (value / y_max if y_max else 0)
            x = start_x + sidx * (bar_width + gap)
            y = 525 - height
            parts.append(f'    <rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" rx="4" fill="{color}"/>')
            parts.append(f'    <text x="{x + bar_width / 2:.1f}" y="{max(102, y - 8):.1f}" text-anchor="middle">{value:.3f}</text>')
        parts.append(f'    <text x="{center:.1f}" y="560" text-anchor="middle" font-size="17">{labels[idx]}</text>')
    parts.append("  </g>")
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_html(output_dir: Path, labels: list[str], data: list[dict]) -> None:
    rows = []
    for label, item in zip(labels, data):
        rows.append(
            f"<tr><td>{label}</td><td>{item['a_win']:.4f}</td><td>{item['b_win']:.4f}</td><td>{item['tie']:.4f}</td>"
            f"<td>{item['a_reward']:.4f}</td><td>{item['b_reward']:.4f}</td><td>{item['scope']}</td></tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CrossPlay start vs checkpoint</title>
  <style>
    body {{ margin: 0; background: #f7f4ec; color: #1f2937; font: 16px/1.5 Georgia, serif; }}
    main {{ max-width: 1120px; margin: 32px auto; padding: 0 20px; }}
    h1 {{ font-size: 30px; margin: 0 0 18px; }}
    p {{ color: #6b7280; margin: 0 0 20px; }}
    img {{ width: 100%; background: #fffdf8; border: 1px solid #ddd6c8; margin: 12px 0 22px; }}
    table {{ border-collapse: collapse; width: 100%; background: #fffdf8; font-family: Arial, sans-serif; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child, td:last-child {{ text-align: left; }}
  </style>
</head>
<body>
<main>
  <h1>CrossPlay start vs checkpoint comparison</h1>
  <p>These panels compare the available initial run summary and the latest successful checkpoint evaluation. The two points use different validation scopes.</p>
  <img src="crossplay_start_vs_ckpt_rates.svg" alt="Outcome rates">
  <img src="crossplay_start_vs_ckpt_rewards.svg" alt="Reward means">
  <table>
    <thead><tr><th>Run</th><th>A win</th><th>B win</th><th>Tie</th><th>A reward</th><th>B reward</th><th>Scope</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_payload = load_json(Path(args.init_json))
    ckpt_payload = load_json(Path(args.ckpt_json))

    labels = ["Init", args.checkpoint_label]
    data = [
        {
            "a_win": metric(init_payload, "val/policy_a_win_rate"),
            "b_win": metric(init_payload, "val/policy_b_win_rate"),
            "tie": metric(init_payload, "val/tie_rate"),
            "a_reward": metric(init_payload, "val/policy_a_reward_mean"),
            "b_reward": metric(init_payload, "val/policy_b_reward_mean"),
            "scope": f"{init_payload.get('max_samples', 'unknown')} sample prefix",
        },
        {
            "a_win": metric(ckpt_payload, "val/policy_a_win_rate"),
            "b_win": metric(ckpt_payload, "val/policy_b_win_rate"),
            "tie": metric(ckpt_payload, "val/tie_rate"),
            "a_reward": metric(ckpt_payload, "val/policy_a_reward_mean"),
            "b_reward": metric(ckpt_payload, "val/policy_b_reward_mean"),
            "scope": "full validation set",
        },
    ]

    note = "Init uses the saved 500-sample prefix; checkpoint uses full GSM8K validation"
    render_grouped_bars(
        output_dir / "crossplay_start_vs_ckpt_rates.svg",
        "CrossPlay outcome rates: start vs checkpoint",
        note,
        labels,
        data,
        [
            ("Win rate(A)", "a_win", COLORS["a"]),
            ("Win rate(B)", "b_win", COLORS["b"]),
            ("Tie rate", "tie", COLORS["tie"]),
        ],
        y_max=1.0,
    )
    render_grouped_bars(
        output_dir / "crossplay_start_vs_ckpt_rewards.svg",
        "CrossPlay reward means: start vs checkpoint",
        note,
        labels,
        data,
        [
            ("E[r_A]", "a_reward", COLORS["a"]),
            ("E[r_B]", "b_reward", COLORS["b"]),
        ],
        y_max=0.2,
    )
    render_html(output_dir, labels, data)


if __name__ == "__main__":
    main()
