import argparse
import csv
import os
import sys
from itertools import product
from pathlib import Path
from typing import Iterable, List, Tuple

from compare_paths import compute_metrics, ensure_backup, ensure_mpl_env


def run_rrt(
    root: Path,
    step_size: float,
    goal_rate: float,
    clearance: float,
    max_iters: int,
    plot: bool,
):
    """Run the RRT planner once with the given parameters."""
    python_dir = root / "python-files"
    cmd = [
        sys.executable,
        str(python_dir / "rrt_planner.py"),
        "--max-iters",
        str(max_iters),
        "--step-size",
        str(step_size),
        "--goal-rate",
        str(goal_rate),
        "--clearance",
        str(clearance),
    ]
    if plot:
        cmd.append("--plot")
    env = ensure_mpl_env(os.environ, root)
    print(f"\n>> Running: {' '.join(cmd)}")
    sys.stdout.flush()
    # spawnve keeps the planner's stdout/stderr streaming to the console
    result = os.spawnve(os.P_WAIT, cmd[0], cmd, env)
    if result != 0:
        raise RuntimeError(f"Planner failed with exit code {result}")


def write_header(writer: csv.writer, include_hybrid: bool):
    header = [
        "step_size",
        "goal_rate",
        "clearance",
        "max_iters",
        "rrt_nodes",
        "rrt_length_m",
    ]
    if include_hybrid:
        header.extend(["hybrid_nodes", "hybrid_length_m"])
    writer.writerow(header)


def main(argv: List[str] = None):
    parser = argparse.ArgumentParser(
        description="Sweep RRT parameters and log path metrics to CSV."
    )
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        default=[3.5],
        help="List of step sizes to test (space separated).",
    )
    parser.add_argument(
        "--goal-rates",
        type=float,
        nargs="+",
        default=[0.3],
        help="List of goal sampling rates to test (space separated).",
    )
    parser.add_argument(
        "--clearances",
        type=float,
        nargs="+",
        default=[0.8],
        help="List of obstacle clearances to test (space separated).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        nargs="+",
        default=[50000],
        help="List of max iteration counts to test (space separated).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sweep_metrics.csv"),
        help="CSV file to write results into (relative to repo root).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the CSV instead of overwriting.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show matplotlib plot for each run (slower).",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    rrt_path = root / "python-files" / "rrt_path.json"

    # Ensure Hybrid A* backup exists and capture its metrics once.
    hybrid_backup = ensure_backup(root)
    hybrid_metrics = compute_metrics(hybrid_backup)

    mode = "a" if args.append and args.output.exists() else "w"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            write_header(writer, include_hybrid=hybrid_metrics is not None)

        combos: Iterable[Tuple[float, float, float, int]] = product(
            args.step_sizes, args.goal_rates, args.clearances, args.max_iters
        )
        for step_size, goal_rate, clearance, max_iters in combos:
            run_rrt(root, step_size, goal_rate, clearance, max_iters, args.plot)
            rrt_metrics = compute_metrics(rrt_path)
            if rrt_metrics is None:
                print(
                    f"Missing RRT path after run (step={step_size}, goal={goal_rate}, "
                    f"clearance={clearance}, iters={max_iters})"
                )
                continue

            row = [
                step_size,
                goal_rate,
                clearance,
                max_iters,
                rrt_metrics[0],
                f"{rrt_metrics[1]:.2f}",
            ]
            if hybrid_metrics is not None:
                row.extend([hybrid_metrics[0], f"{hybrid_metrics[1]:.2f}"])
            writer.writerow(row)
            print(
                f"Logged step={step_size}, goal={goal_rate}, clear={clearance}, "
                f"iters={max_iters} -> nodes={rrt_metrics[0]}, "
                f"length={rrt_metrics[1]:.2f} m"
            )


if __name__ == "__main__":
    main()
