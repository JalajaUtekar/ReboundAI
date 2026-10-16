import json
import subprocess
import sys
from pathlib import Path


def test_cli_writes_valid_json_dataset(tmp_path):
    output_path = tmp_path / "synthetic_students.json"
    result = subprocess.run(
        [sys.executable, "scripts/generate_synthetic_data.py", "--output", str(output_path), "--seed", "42"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 75
    assert len(data[0]["subjects"]) == 6
