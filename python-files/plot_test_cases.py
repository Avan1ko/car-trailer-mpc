"""
Quick sketch visualizations for the poses defined in test_cases.json.

For each case we draw:
- Start pose (green arrow) with heading.
- Goal pose (blue arrow) with heading.
- A dashed straight-line connector as a rough intended path sketch.

Images are saved under _media/test_case_sketches/<case>.png.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent.parent
TEST_CASES_PATH = ROOT / "test_cases.json"
OUTPUT_DIR = ROOT / "_media" / "test_case_sketches"


@dataclass
class Pose:
    x: float
    y: float
    heading_rad: float

    @classmethod
    def from_dict(cls, data: dict) -> "Pose":
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            heading_rad=float(data["heading_rad"]),
        )


@dataclass
class TestCase:
    name: str
    start: Pose
    goal: Pose
    notes: str

    @classmethod
    def from_dict(cls, data: dict) -> "TestCase":
        return cls(
            name=data["name"],
            start=Pose.from_dict(data["start"]),
            goal=Pose.from_dict(data["goal"]),
            notes=data.get("notes", ""),
        )


def load_cases(path: Path) -> Tuple[str, Tuple[TestCase, ...]]:
    raw = json.loads(path.read_text())
    unit_info = raw.get("units", {})
    cases = tuple(TestCase.from_dict(c) for c in raw["cases"])
    return unit_info, cases


def draw_pose(ax, pose: Pose, color: str, label: str, arrow_scale: float = 3.0):
    dx = math.cos(pose.heading_rad) * arrow_scale
    dy = math.sin(pose.heading_rad) * arrow_scale
    ax.arrow(
        pose.x,
        pose.y,
        dx,
        dy,
        head_width=1.4,
        head_length=2.1,
        length_includes_head=True,
        color=color,
        linewidth=2.0,
    )
    ax.text(pose.x, pose.y, label, color=color, fontsize=9, weight="bold", ha="center", va="center")


def draw_case(case: TestCase):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(case.name)

    draw_pose(ax, case.start, color="green", label="S")
    draw_pose(ax, case.goal, color="royalblue", label="G")

    # Straight-line sketch for intended motion
    ax.plot(
        [case.start.x, case.goal.x],
        [case.start.y, case.goal.y],
        linestyle="--",
        color="gray",
        linewidth=1.5,
    )

    # Add notes if available
    if case.notes:
        ax.annotate(
            case.notes,
            xy=(0.5, -0.12),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=9,
            color="dimgray",
            wrap=True,
        )

    # Pad view to keep arrows and text clear
    xs = [case.start.x, case.goal.x]
    ys = [case.start.y, case.goal.y]
    pad = 6.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=8)
    start_handle = Line2D([0], [0], color="green", marker=">", linestyle="None", markersize=10, label="start")
    goal_handle = Line2D([0], [0], color="royalblue", marker=">", linestyle="None", markersize=10, label="goal")
    ax.legend(handles=[start_handle, goal_handle], loc="upper right")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{case.name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    unit_info, cases = load_cases(TEST_CASES_PATH)
    print(f"Loaded {len(cases)} cases; units: {unit_info}")
    for case in cases:
        out = draw_case(case)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
