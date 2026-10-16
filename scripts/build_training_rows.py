"""CLI entrypoint: build Rebound AI's no-leakage ScholarTrack training rows.

Reads the generated synthetic dataset (data/synthetic_students.json by default),
extracts one no-leakage prefix row per (student, subject, valid as-of checkpoint),
and writes both a JSON and a flattened CSV export.

Usage:
    python scripts/build_training_rows.py [--input data/synthetic_students.json]
                                           [--output-json data/training_rows.json]
                                           [--output-csv data/training_rows.csv]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.feature_engineering.extraction import FEATURE_NAMES, build_training_rows

ROW_COLUMNS = ["student_id", "subject_slug", "checkpoint_idx", *FEATURE_NAMES, "at_risk", "is_failing"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Rebound AI's no-leakage ScholarTrack training rows.")
    parser.add_argument("--input", default="data/synthetic_students.json")
    parser.add_argument("--output-json", default="data/training_rows.json")
    parser.add_argument("--output-csv", default="data/training_rows.csv")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        dataset = json.load(f)

    rows = build_training_rows(dataset)

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    at_risk_count = sum(1 for r in rows if r["at_risk"])
    failing_count = sum(1 for r in rows if r["is_failing"])
    print(
        f"Wrote {len(rows)} training rows to {json_path} and {csv_path} "
        f"({at_risk_count} at_risk rows, {failing_count} is_failing rows)."
    )


if __name__ == "__main__":
    main()
