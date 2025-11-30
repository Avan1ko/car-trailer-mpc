import argparse
import json
from pathlib import Path


def build_obstacles(open_spot: int, depth: float = 20.0):
    """
    Build one-sided parking obstacles across 10 adjacent stalls.
    We leave exactly one stall (1-based index) open.
    """
    if not 1 <= open_spot <= 10:
        raise ValueError("open_spot must be between 1 and 10 (inclusive)")

    obstacles = []
    stall_width = 10.0
    stripe_width = 1.0
    x_start = stripe_width  # first open gap starts after stripe at x=0..1

    for idx in range(10):
        if (idx + 1) == open_spot:
            continue  # leave this stall open
        x0 = x_start + idx * (stall_width + stripe_width)
        x1 = x0 + stall_width
        obstacles.append(
            {
                "FL": {"X": x0, "Y": depth},
                "FR": {"X": x1, "Y": depth},
                "BL": {"X": x0, "Y": 0.0},
                "BR": {"X": x1, "Y": 0.0},
            }
        )
    return obstacles


def main():
    parser = argparse.ArgumentParser(
        description="Generate obstacles.json with all stalls blocked except one open slot."
    )
    parser.add_argument(
        "--open-spot",
        type=int,
        default=4,
        help="1-based stall index (1..10) to leave open.",
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=20.0,
        help="Depth of each stall in meters (Y direction).",
    )
    parser.add_argument(
        "--update-test-case",
        action="store_true",
        help="Also update a test case goal to the center of the open stall.",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default="left_offset_reverse_turn_in",
        help="Test case name to update when --update-test-case is set.",
    )
    parser.add_argument(
        "--write-initialize",
        action="store_true",
        help="Also move the goal in initialize.json to the open stall center.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "obstacles.json",
        help="Where to write the generated obstacles.json",
    )
    args = parser.parse_args()

    obstacles = build_obstacles(args.open_spot, args.depth)
    args.output.write_text(json.dumps(obstacles, indent=2))
    print(f"Wrote {len(obstacles)} obstacles to {args.output} (open spot {args.open_spot})")

    if args.update_test_case or args.write_initialize:
        stall_width = 10.0
        stripe_width = 1.0
        x0 = stripe_width + (args.open_spot - 1) * (stall_width + stripe_width)
        goal_x = x0 + stall_width / 2.0
        goal_y = args.depth / 2.0  # center of the bay

    if args.update_test_case:
        tc_path = Path(__file__).resolve().parent.parent / "test_cases.json"
        data = json.loads(tc_path.read_text())
        updated = False
        for case in data.get("cases", []):
            if case.get("name") == args.case_name:
                case["goal"]["x"] = goal_x
                case["goal"]["y"] = goal_y
                updated = True
                break
        if updated:
            tc_path.write_text(json.dumps(data, indent=2))
            print(f"Updated {args.case_name} goal in {tc_path} to x={goal_x:.2f}, y={goal_y:.2f}")
        else:
            print(f"Case {args.case_name} not found in {tc_path}; no goal updated.")

    if args.write_initialize:
        init_path = Path(__file__).resolve().parent.parent / "initialize.json"
        init_data = json.loads(init_path.read_text())
        if "Positions" in init_data and len(init_data["Positions"]) > 0:
            init_data["Positions"][-1] = [goal_x, goal_y]
        else:
            init_data["Positions"] = [[goal_x, goal_y]]
        init_path.write_text(json.dumps(init_data, indent=2))
        print(f"Updated initialize.json goal to x={goal_x:.2f}, y={goal_y:.2f}")


if __name__ == "__main__":
    main()
