import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def ensure_backup(root: Path) -> Path:
    initialize = root / "initialize.json"
    backup = root / "initialize_hybrid.json"
    if not backup.exists() and initialize.exists():
        shutil.copy(initialize, backup)
        print(f"Backed up baseline path -> {backup}")
    return backup


def compute_metrics(path_file: Path):
    if not path_file.exists():
        return None
    with open(path_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    positions = np.asarray(data.get("Positions", []), dtype=float)
    if len(positions) < 2:
        return len(positions), 0.0
    diffs = positions[1:] - positions[:-1]
    total_length = float(np.linalg.norm(diffs, axis=1).sum())
    return len(positions), total_length


def ensure_mpl_env(env: dict, workdir: Path) -> dict:
    env = env.copy()
    if "MPLCONFIGDIR" not in env:
        mpl_dir = workdir / ".mplconfig"
        mpl_dir.mkdir(exist_ok=True)
        env["MPLCONFIGDIR"] = str(mpl_dir)
    return env


def run_command(cmd, cwd: Path):
    env = ensure_mpl_env(os.environ, cwd)
    print(f"\n>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def summarize(label: str, path_file: Path):
    metrics = compute_metrics(path_file)
    if metrics is None:
        print(f"{label}: missing file {path_file}")
        return None
    nodes, length = metrics
    print(f"{label} -> nodes: {nodes}, path length: {length:.2f} m")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare Hybrid A* and RRT path metrics.")
    parser.add_argument("--step-size", type=float, default=3.5)
    parser.add_argument("--max-iters", type=int, default=50000)
    parser.add_argument("--goal-rate", type=float, default=0.3)
    parser.add_argument("--clearance", type=float, default=0.8)
    parser.add_argument("--plot", action="store_true", help="show the RRT plot during planning")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    python_dir = root / "python-files"

    hybrid_backup = ensure_backup(root)
    summarize("Hybrid A*", hybrid_backup)

    planner_cmd = [
        sys.executable,
        str(python_dir / "rrt_planner.py"),
        "--max-iters",
        str(args.max_iters),
        "--step-size",
        str(args.step_size),
        "--goal-rate",
        str(args.goal_rate),
        "--clearance",
        str(args.clearance),
    ]
    if args.plot:
        planner_cmd.append("--plot")
    run_command(planner_cmd, root)

    rrt_path = python_dir / "rrt_path.json"
    summarize("RRT", rrt_path)


if __name__ == "__main__":
    main()
