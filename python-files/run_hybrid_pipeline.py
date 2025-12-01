import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def write_seed_initialize(start, goal, output: Path):
    data = {
        "Positions": [
            [start[0], start[1]],
            [goal[0], goal[1]],
        ],
        "Headings": [
            start[2],
            goal[2],
        ],
        "HitchAngles": [
            start[3],
            goal[3],
        ],
    }
    output.write_text(json.dumps(data, indent=2))
    print(f"Wrote seed initialize.json -> {output}")


def main():
    parser = argparse.ArgumentParser(description="Write start/goal, run Hybrid A* seed, then trajectory optimization.")
    parser.add_argument("--start", type=float, nargs=4, metavar=("X", "Y", "HEADING", "HITCH"), required=True)
    parser.add_argument("--goal", type=float, nargs=4, metavar=("X", "Y", "HEADING", "HITCH"), required=True)
    parser.add_argument("--no-optimize", action="store_true", help="Skip running trajectory_animation.py")
    parser.add_argument("--guess-output", type=str, default="initialize_guess.json", help="Where to write the Hybrid A* path.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    initialize = root / "initialize.json"
    guess_path = root / args.guess_output

    write_seed_initialize(args.start, args.goal, initialize)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))

    # Run Hybrid A* to populate guess file with a path
    ha_cmd = [
        sys.executable,
        str(root / "python-files" / "hybrid_astar_planner.py"),
        "--output",
        str(guess_path),
    ]
    print(f"\n>> Running Hybrid A*: {' '.join(ha_cmd)}")
    ha_proc = subprocess.run(ha_cmd, cwd=root, env=env)
    if ha_proc.returncode != 0:
        print("Hybrid A* failed or returned non-zero; keeping the seed start/goal only.")

    if args.no_optimize:
        return

    # Run trajectory optimization + animation
    anim_cmd = [sys.executable, str(root / "python-files" / "trajectory_animation.py")]
    print(f"\n>> Running trajectory_animation: {' '.join(anim_cmd)}")
    subprocess.run(anim_cmd, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
