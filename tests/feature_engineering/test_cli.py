import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_writes_json_and_csv_training_rows(tmp_path):
    output_json = tmp_path / "training_rows.json"
    output_csv = tmp_path / "training_rows.csv"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_training_rows.py",
            "--input",
            "data/synthetic_students.json",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    with open(output_json, encoding="utf-8") as f:
        rows = json.load(f)
    assert len(rows) == 4320

    with open(output_csv, newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 4320
    assert "marks_deficit_margin" in csv_rows[0]
    assert "at_risk" in csv_rows[0]
    assert "is_failing" in csv_rows[0]
