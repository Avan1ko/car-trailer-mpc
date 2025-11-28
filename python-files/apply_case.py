import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_CASES = ROOT / "test_cases.json"
DEFAULT_OUTPUT = ROOT / "initialize.json"


def load_cases() -> dict:
    with TEST_CASES.open() as f:
        payload = json.load(f)
    return {case["name"]: case for case in payload["cases"]}


def write_initialize(case: dict, output_path: Path) -> None:
    start = case["start"]
    goal = case["goal"]
    doc = {
        "Positions": [
            [start["x"], start["y"]],
            [goal["x"], goal["y"]],
        ],
        "Headings": [
            start["heading_rad"],
            goal["heading_rad"],
        ],
        "HitchAngles": [
            start["hitch_angle_rad"],
            goal["hitch_angle_rad"],
        ],
    }
    with output_path.open("w") as f:
        json.dump(doc, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a named test case to initialize.json")
    parser.add_argument("--case", required=True, help="Case name from test_cases.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the initialize-style JSON (default: initialize.json in repo root)",
    )
    args = parser.parse_args()

    cases = load_cases()
    if args.case not in cases:
        available = ", ".join(sorted(cases.keys()))
        raise SystemExit(f"Unknown case '{args.case}'. Available: {available}")

    write_initialize(cases[args.case], args.output)
    print(f"Wrote '{args.case}' to {args.output}")


if __name__ == "__main__":
    main()
