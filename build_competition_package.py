"""Bundle competition submission into single CSV/JSON files."""
import json
import csv
from pathlib import Path

SUBMISSION_DIR = Path("results/submission")


def bundle_scenario(scenario):
    src_dir = SUBMISSION_DIR / f"scenario{scenario}"
    if not src_dir.exists():
        print(f"Scenario {scenario} directory not found: {src_dir}")
        return

    files = sorted(src_dir.glob("*.json"))
    print(f"Scenario {scenario}: found {len(files)} instance files")

    all_runs = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            all_runs.extend(data)
        else:
            all_runs.append(data)

    out_json = SUBMISSION_DIR / f"scenario{scenario}_combined.json"
    with open(out_json, 'w') as f:
        json.dump(all_runs, f, indent=2)
    print(f"  -> {out_json} ({len(all_runs)} runs)")

    out_csv = SUBMISSION_DIR / f"scenario{scenario}_combined.csv"
    if all_runs:
        keys = list(all_runs[0].keys())
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_runs)
        print(f"  -> {out_csv}")


if __name__ == "__main__":
    bundle_scenario(1)
    bundle_scenario(2)
    print("\nDone.")