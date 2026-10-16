"""CLI entrypoint: regenerate Rebound AI's synthetic ScholarTrack dataset.

Usage:
    python scripts/generate_synthetic_data.py [--output data/synthetic_students.json] [--seed 42]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.data_generation.generator import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Rebound AI's synthetic student dataset.")
    parser.add_argument("--output", default="data/synthetic_students.json")
    parser.add_argument("--num-students", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = generate_dataset(num_students=args.num_students, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    at_risk_count = sum(1 for s in dataset if any(subj["outcome"]["at_risk"] for subj in s["subjects"]))
    failing_count = sum(1 for s in dataset if any(subj["outcome"]["is_failing"] for subj in s["subjects"]))
    print(
        f"Wrote {len(dataset)} students to {output_path} "
        f"({at_risk_count} flagged at-risk, {failing_count} failing)."
    )


if __name__ == "__main__":
    main()
