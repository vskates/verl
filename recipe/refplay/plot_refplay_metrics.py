import argparse
import csv
import re
from pathlib import Path


STEP_RE = re.compile(r"step:(?P<step>\d+)\s+-\s+(?P<body>.*)")
METRIC_RE = re.compile(r"(?P<key>[A-Za-z0-9_./-]+):(?P<value>-?\d+(?:\.\d+)?)")

DEFAULT_METRICS = [
    "game/actor_reward_mean",
    "game/reference_reward_mean",
    "game/actor_win_rate",
    "game/reward_margin_mean",
    "actor/dpo_loss",
    "time/step",
]


def moving_average(values, window: int):
    if window <= 1 or len(values) <= 2:
        return list(values)

    half_window = window // 2
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)
        chunk = values[start:end]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def parse_log(log_path: Path):
    rows = []
    for line in log_path.read_text(errors="ignore").splitlines():
        match = STEP_RE.search(line)
        if match is None:
            continue
        row = {"step": int(match.group("step"))}
        for metric in METRIC_RE.finditer(match.group("body")):
            row[metric.group("key")] = float(metric.group("value"))
        rows.append(row)
    return rows


def write_csv(rows, metrics, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *metrics])
        for row in rows:
            writer.writerow([row["step"], *[row.get(metric, "") for metric in metrics]])


def plot_metrics(rows, metrics, output_dir: Path, smoothing_window: int | None = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render reward plots") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    steps = [row["step"] for row in rows]

    for metric in metrics:
        values = [row.get(metric) for row in rows]
        filtered = [(step, value) for step, value in zip(steps, values) if value is not None]
        if not filtered:
            continue

        metric_steps = [step for step, _ in filtered]
        metric_values = [value for _, value in filtered]
        window = smoothing_window or max(5, len(metric_values) // 25)
        smoothed_values = moving_average(metric_values, window)
        plt.figure(figsize=(8, 4.5))
        plt.plot(
            metric_steps,
            metric_values,
            color="#94a3b8",
            linewidth=1.2,
            alpha=0.35,
        )
        plt.scatter(
            metric_steps,
            metric_values,
            color="#94a3b8",
            s=10,
            alpha=0.22,
            edgecolors="none",
        )
        plt.plot(
            metric_steps,
            smoothed_values,
            color="#2563eb",
            linewidth=2.6,
            alpha=0.98,
            label=f"smoothed (window={window})",
        )
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.title(metric)
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False)
        safe_name = metric.replace("/", "__")
        plt.tight_layout()
        plt.savefig(output_dir / f"{safe_name}.png", dpi=160)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot RefPlay train metrics from a console log.")
    parser.add_argument("log_path", type=Path, help="Path to a refplay train log file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG/CSV outputs")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Metrics to export and plot")
    parser.add_argument("--smoothing-window", type=int, default=None, help="Centered moving-average window for the bright trend line")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.log_path.parent / f"{args.log_path.stem}_plots")
    rows = parse_log(args.log_path)
    if not rows:
        raise SystemExit(f"No step metrics found in {args.log_path}")

    write_csv(rows, args.metrics, output_dir / "metrics.csv")
    plot_metrics(rows, args.metrics, output_dir, smoothing_window=args.smoothing_window)
    print(f"Parsed {len(rows)} steps from {args.log_path}")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
