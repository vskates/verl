import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paired CrossPlay examples into a markdown report.")
    parser.add_argument("--init-json", required=True)
    parser.add_argument("--ckpt-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_examples(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("examples", [])


def trim(text: str, limit: int = 1200) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def render_example_row(idx: int, init_ex: dict, ckpt_ex: dict) -> str:
    prompt = trim(init_ex.get("prompt_text", ""))
    ground_truth = init_ex.get("ground_truth", "")
    return "\n".join(
        [
            f"## Example {idx + 1}",
            "",
            f"- `sample_index`: {init_ex.get('sample_index')}",
            f"- `benchmark_id`: {init_ex.get('benchmark_id')}",
            f"- `ground_truth`: `{ground_truth}`",
            "",
            "### Prompt",
            "",
            "```text",
            prompt,
            "```",
            "",
            "### Before Training",
            "",
            f"- `policy_a_score`: {init_ex.get('policy_a_score', 0.0):.3f}",
            f"- `policy_b_score`: {init_ex.get('policy_b_score', 0.0):.3f}",
            f"- `winner`: `{init_ex.get('winner', 'tie')}`",
            f"- `policy_a_predicted_answer`: `{init_ex.get('policy_a_predicted_answer', '')}`",
            f"- `policy_b_predicted_answer`: `{init_ex.get('policy_b_predicted_answer', '')}`",
            "",
            "```text",
            "[A0]",
            trim(init_ex.get("policy_a_response", "")),
            "",
            "[B0]",
            trim(init_ex.get("policy_b_response", "")),
            "```",
            "",
            "### After Training",
            "",
            f"- `policy_a_score`: {ckpt_ex.get('policy_a_score', 0.0):.3f}",
            f"- `policy_b_score`: {ckpt_ex.get('policy_b_score', 0.0):.3f}",
            f"- `winner`: `{ckpt_ex.get('winner', 'tie')}`",
            f"- `policy_a_predicted_answer`: `{ckpt_ex.get('policy_a_predicted_answer', '')}`",
            f"- `policy_b_predicted_answer`: `{ckpt_ex.get('policy_b_predicted_answer', '')}`",
            "",
            "```text",
            "[A3737]",
            trim(ckpt_ex.get("policy_a_response", "")),
            "",
            "[B3737]",
            trim(ckpt_ex.get("policy_b_response", "")),
            "```",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    init_examples = load_examples(Path(args.init_json))
    ckpt_examples = load_examples(Path(args.ckpt_json))
    count = min(len(init_examples), len(ckpt_examples))

    rows = [
        "# CrossPlay qualitative examples",
        "",
        "Сравнение ответов двух моделей до и после обучения на одних и тех же validation prompts.",
        "",
    ]
    for idx in range(count):
        rows.append(render_example_row(idx, init_examples[idx], ckpt_examples[idx]))

    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
