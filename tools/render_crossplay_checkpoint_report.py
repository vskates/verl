import argparse
import json
import re
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
    parser = argparse.ArgumentParser(description="Render CrossPlay checkpoint eval graphs.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="CrossPlay checkpoint evaluation")
    return parser.parse_args()


def checkpoint_step(path: Path, payload: dict) -> int:
    text = payload.get("experiment_name", "") + " " + path.name
    match = re.search(r"global_step_(\d+)", text)
    if not match:
        raise ValueError(f"Cannot infer checkpoint step from {path}")
    return int(match.group(1))


def metric(payload: dict, name: str) -> float:
    return float(payload["metrics"][name])


def load_results(data_dir: Path) -> list[dict]:
    results = []
    for path in sorted(data_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        step = checkpoint_step(path, payload)
        results.append(
            {
                "step": step,
                "path": path,
                "a_win": metric(payload, "val/policy_a_win_rate"),
                "b_win": metric(payload, "val/policy_b_win_rate"),
                "tie": metric(payload, "val/tie_rate"),
                "a_reward": metric(payload, "val/policy_a_reward_mean"),
                "b_reward": metric(payload, "val/policy_b_reward_mean"),
                "reward_manager": payload.get("reward_manager", ""),
                "max_response_length": payload.get("max_response_length", ""),
            }
        )
    if not results:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
    return sorted(results, key=lambda item: item["step"])


def write_summary(output_dir: Path, results: list[dict]) -> None:
    rows = [
        "checkpoint\tpolicy_a_win_rate\tpolicy_b_win_rate\ttie_rate\tpolicy_a_reward_mean\tpolicy_b_reward_mean"
    ]
    for item in results:
        rows.append(
            "\t".join(
                [
                    str(item["step"]),
                    f"{item['a_win']:.6f}",
                    f"{item['b_win']:.6f}",
                    f"{item['tie']:.6f}",
                    f"{item['a_reward']:.6f}",
                    f"{item['b_reward']:.6f}",
                ]
            )
        )
    (output_dir / "summary.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def svg_start(title: str, subtitle: str, y_max: float) -> list[str]:
    lines = [
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
        lines.append(f'    <line x1="95" y1="{y:.1f}" x2="1010" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')
        lines.append(f'    <text x="82" y="{y + 5:.1f}" text-anchor="end">{value:.2f}</text>')
    lines.extend(["  </g>", ""])
    return lines


def render_grouped_bars(
    output_path: Path,
    title: str,
    subtitle: str,
    results: list[dict],
    series: list[tuple[str, str, str]],
    y_max: float,
) -> None:
    parts = svg_start(title, subtitle, y_max)
    legend_x = 730
    parts.append(f'  <g font-family="Arial, sans-serif" font-size="15" fill="{COLORS["text"]}">')
    for idx, (label, _, color) in enumerate(series):
        y = 96 + 28 * idx
        parts.append(f'    <rect x="{legend_x}" y="{y}" width="16" height="16" rx="3" fill="{color}"/>')
        parts.append(f'    <text x="{legend_x + 26}" y="{y + 13}">{label}</text>')
    parts.extend(["  </g>", ""])

    plot_left = 145
    plot_width = 805
    group_width = plot_width / max(1, len(results))
    bar_width = min(42, group_width / (len(series) + 1.8))
    gap = bar_width * 0.35

    parts.append(f'  <g font-family="Arial, sans-serif" font-size="13" fill="{COLORS["text"]}">')
    for idx, item in enumerate(results):
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
        parts.append(f'    <text x="{center:.1f}" y="560" text-anchor="middle" font-size="17">step {item["step"]}</text>')
    parts.append("  </g>")
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_html(output_dir: Path, results: list[dict], title: str) -> None:
    rows = []
    for item in results:
        rows.append(
            f"<tr><td>{item['step']}</td><td>{item['a_win']:.4f}</td><td>{item['b_win']:.4f}</td>"
            f"<td>{item['tie']:.4f}</td><td>{item['a_reward']:.4f}</td><td>{item['b_reward']:.4f}</td></tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin: 0; background: #f7f4ec; color: #1f2937; font: 16px/1.5 Georgia, serif; }}
    main {{ max-width: 1120px; margin: 32px auto; padding: 0 20px; }}
    h1 {{ font-size: 30px; margin: 0 0 18px; }}
    img {{ width: 100%; background: #fffdf8; border: 1px solid #ddd6c8; margin: 12px 0 22px; }}
    table {{ border-collapse: collapse; width: 100%; background: #fffdf8; font-family: Arial, sans-serif; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <img src="crossplay_ckpt_rates.svg" alt="Outcome rates by checkpoint">
  <img src="crossplay_ckpt_rewards.svg" alt="Reward means by checkpoint">
  <table>
    <thead><tr><th>Checkpoint</th><th>A win</th><th>B win</th><th>Tie</th><th>A reward</th><th>B reward</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(data_dir)
    write_summary(output_dir, results)
    subtitle = "A = Qwen2.5-0.5B, B = SmolLM2-1.7B, full GSM8K validation"

    render_grouped_bars(
        output_dir / "crossplay_ckpt_rates.svg",
        "CrossPlay outcome rates by checkpoint",
        subtitle,
        results,
        [
            ("Win rate(A)", "a_win", COLORS["a"]),
            ("Win rate(B)", "b_win", COLORS["b"]),
            ("Tie rate", "tie", COLORS["tie"]),
        ],
        y_max=0.6,
    )
    render_grouped_bars(
        output_dir / "crossplay_ckpt_rewards.svg",
        "CrossPlay reward means by checkpoint",
        subtitle,
        results,
        [
            ("E[r_A]", "a_reward", COLORS["a"]),
            ("E[r_B]", "b_reward", COLORS["b"]),
        ],
        y_max=0.2,
    )
    render_html(output_dir, results, args.title)


if __name__ == "__main__":
    main()
