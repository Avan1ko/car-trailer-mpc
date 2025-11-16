import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_mpl_config(env: dict, workdir: Path) -> dict:
    """Make sure matplotlib has somewhere writable for cache/config."""
    env = env.copy()
    if "MPLCONFIGDIR" not in env:
        mpl_dir = workdir / ".mplconfig"
        mpl_dir.mkdir(exist_ok=True)
        env["MPLCONFIGDIR"] = str(mpl_dir)
    return env


def run_command(cmd, cwd: Path):
    env = ensure_mpl_config(os.environ, cwd)
    print(f"\n>> Running: {' '.join(cmd)} (cwd={cwd})")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def backup_initialize(root: Path, backup_name: str) -> Path:
    initialize = root / "initialize.json"
    backup_path = root / backup_name
    if initialize.exists():
        shutil.copy(initialize, backup_path)
        print(f"Backed up initialize.json -> {backup_path}")
    return backup_path


def apply_rrt_path(root: Path, rrt_output: Path, backup: bool):
    if backup:
        backup_initialize(root, "initialize_hybrid.json")
    target = root / "initialize.json"
    shutil.copy(rrt_output, target)
    print(f"Copied {rrt_output} -> {target}")


def restore_backup(root: Path, backup_name: str = "initialize_hybrid.json"):
    backup = root / backup_name
    target = root / "initialize.json"
    if backup.exists():
        shutil.copy(backup, target)
        print(f"Restored {backup} -> {target}")
    else:
        print(f"No backup {backup} found; skipping restore.")


def main():
    parser = argparse.ArgumentParser(
        description="Run the planar RRT planner and optionally the full workflow."
    )
    parser.add_argument("--step-size", type=float, default=3.5)
    parser.add_argument("--max-iters", type=int, default=50000)
    parser.add_argument("--goal-rate", type=float, default=0.3)
    parser.add_argument("--clearance", type=float, default=0.8)
    parser.add_argument("--output", type=str, default="rrt_path.json")
    parser.add_argument("--plot", action="store_true", help="Show RRT plot when planning")
    parser.add_argument("--apply", action="store_true", help="Copy RRT path into initialize.json")
    parser.add_argument("--no-backup", action="store_true", help="Skip backing up initialize.json when applying")
    parser.add_argument("--optimize", action="store_true", help="Run trajectory_animation.py after planning")
    parser.add_argument("--simulate", action="store_true", help="Run simulation.py after planning/optimization")
    parser.add_argument("--restore", action="store_true", help="Restore initialize.json from initialize_hybrid.json at the end")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    python_dir = root / "python-files"
    output_path = python_dir / args.output

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
        "--output",
        str(output_path),
    ]
    if args.plot:
        planner_cmd.append("--plot")
    run_command(planner_cmd, root)

    if args.apply:
        apply_rrt_path(root, output_path, backup=not args.no_backup)

    if args.optimize:
        run_command([sys.executable, str(python_dir / "trajectory_animation.py")], root)

    if args.simulate:
        run_command([sys.executable, str(python_dir / "simulation.py")], root)

    if args.restore:
        restore_backup(root)

    print("\nRRT workflow completed.")


if __name__ == "__main__":
    main()
