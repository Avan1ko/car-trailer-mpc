import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from get_initial_goal_states import get_initial_goal_states
from get_obstacles import get_obstacles


@dataclass
class AxisAlignedRectangle:
    center: Tuple[float, float]
    width: float
    height: float

    def bounds(self) -> Tuple[float, float, float, float]:
        cx, cy = self.center
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        return (cx - half_w, cx + half_w, cy - half_h, cy + half_h)

    def contains(self, point: np.ndarray, clearance: float = 0.0) -> bool:
        x_min, x_max, y_min, y_max = self.bounds()
        return (
            x_min - clearance <= point[0] <= x_max + clearance
            and y_min - clearance <= point[1] <= y_max + clearance
        )


class Node:
    def __init__(self, point: np.ndarray, parent: Optional[int]):
        self.point = point
        self.parent = parent


class PlanarRRTPlanner:
    """Basic geometric RRT in the warehouse map."""

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        obstacles: Sequence[AxisAlignedRectangle],
        step_size: float = 2.0,
        max_iterations: int = 5000,
        goal_sample_rate: float = 0.1,
        clearance: float = 2.0,
    ):
        self.bounds = bounds
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.clearance = clearance

    def plan(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        nodes: List[Node] = [Node(start, None)]
        for iteration in range(self.max_iterations):
            rnd = self._sample(goal)
            nearest_idx = self._nearest(nodes, rnd)
            new_point = self._steer(nodes[nearest_idx].point, rnd)
            if not self._is_in_bounds(new_point):
                continue
            if not self._segment_is_collision_free(nodes[nearest_idx].point, new_point):
                continue
            nodes.append(Node(new_point, nearest_idx))

            if np.linalg.norm(new_point - goal) <= self.step_size:
                if self._segment_is_collision_free(new_point, goal):
                    nodes.append(Node(goal, len(nodes) - 1))
                    print(f"RRT goal reached after {iteration+1} iterations.")
                    return self._reconstruct(nodes, len(nodes) - 1)

        raise RuntimeError("RRT failed to find a path within iteration limit.")

    def _sample(self, goal: np.ndarray) -> np.ndarray:
        if random.random() < self.goal_sample_rate:
            return goal
        x = random.uniform(*self.bounds["x"])
        y = random.uniform(*self.bounds["y"])
        return np.array([x, y])

    @staticmethod
    def _nearest(nodes: Sequence[Node], point: np.ndarray) -> int:
        distances = [np.linalg.norm(node.point - point) for node in nodes]
        return int(np.argmin(distances))

    def _steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return from_point.copy()
        direction /= distance
        step = min(self.step_size, distance)
        return from_point + step * direction

    def _is_in_bounds(self, point: np.ndarray) -> bool:
        return (
            self.bounds["x"][0] <= point[0] <= self.bounds["x"][1]
            and self.bounds["y"][0] <= point[1] <= self.bounds["y"][1]
        )

    def _segment_is_collision_free(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        distance = np.linalg.norm(p1 - p0)
        steps = max(2, int(distance / (self.step_size / 2.0)))
        for s in range(steps + 1):
            t = s / steps
            point = (1 - t) * p0 + t * p1
            if self._collides(point):
                return False
        return True

    def _collides(self, point: np.ndarray) -> bool:
        for obs in self.obstacles:
            if obs.contains(point, clearance=self.clearance):
                return True
        return False

    def _reconstruct(self, nodes: Sequence[Node], goal_idx: int) -> List[np.ndarray]:
        order: List[int] = []
        idx: Optional[int] = goal_idx
        while idx is not None:
            order.append(idx)
            idx = nodes[idx].parent
        order.reverse()
        return [nodes[i].point for i in order]


def build_obstacle_list(raw_obstacles: Sequence[Dict[str, object]]) -> List[AxisAlignedRectangle]:
    rects = []
    for obs in raw_obstacles:
        center = obs["center"]
        width = obs["width"]
        height = obs["height"]
        rects.append(
            AxisAlignedRectangle(
                center=(float(center[0]), float(center[1])),
                width=float(width),
                height=float(height),
            )
        )
    return rects


def compute_bounds(obstacles: Sequence[AxisAlignedRectangle], start: np.ndarray, goal: np.ndarray) -> Dict[str, Tuple[float, float]]:
    xs = [start[0], goal[0]]
    ys = [start[1], goal[1]]
    for obs in obstacles:
        x_min, x_max, y_min, y_max = obs.bounds()
        xs.extend([x_min, x_max])
        ys.extend([y_min, y_max])
    margin = 5.0
    return {
        "x": (min(xs) - margin, max(xs) + margin),
        "y": (min(ys) - margin, max(ys) + margin),
    }


def convert_points_to_states(points: Sequence[np.ndarray]) -> List[List[float]]:
    headings: List[float] = []
    for idx in range(len(points)):
        if idx < len(points) - 1:
            delta = points[idx + 1] - points[idx]
        else:
            delta = points[idx] - points[idx - 1]
        headings.append(math.atan2(delta[1], delta[0]) - math.pi / 2.0)
    states = []
    for idx, point in enumerate(points):
        heading = headings[idx]
        states.append(
            [
                float(point[0]),
                float(point[1]),
                heading,
                0.0,  # hitch angle (set to zero)
            ]
        )
    return states


def save_to_json(states: Sequence[List[float]], output_path: str) -> None:
    positions = [[state[0], state[1]] for state in states]
    headings = [state[2] for state in states]
    hitch_angles = [state[3] for state in states]
    data = {
        "Positions": positions,
        "Headings": headings,
        "HitchAngles": hitch_angles,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def summarize_path(points: Sequence[np.ndarray]) -> None:
    if len(points) < 2:
        print("RRT summary: insufficient points.")
        return
    total_length = 0.0
    for i in range(len(points) - 1):
        total_length += np.linalg.norm(points[i + 1] - points[i])
    print(
        f"RRT summary -> nodes: {len(points)}, "
        f"path length: {total_length:.2f} m"
    )


def plot_path(points: Sequence[np.ndarray], obstacles: Sequence[AxisAlignedRectangle]) -> None:
    plt.figure()
    for obs in obstacles:
        x_min, x_max, y_min, y_max = obs.bounds()
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor="gray",
            alpha=0.5,
        )
        plt.gca().add_patch(rect)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.plot(xs, ys, "-o", label="RRT path")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Planar RRT path")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Planar RRT planner for the warehouse map.")
    parser.add_argument("--step-size", type=float, default=2.0, help="Step size between nodes.")
    parser.add_argument("--max-iters", type=int, default=5000, help="Maximum RRT iterations.")
    parser.add_argument("--goal-rate", type=float, default=0.1, help="Probability of sampling the goal.")
    parser.add_argument("--clearance", type=float, default=2.0, help="Clearance around obstacle rectangles.")
    parser.add_argument("--output", type=str, default="rrt_path.json", help="Output JSON path.")
    parser.add_argument("--plot", action="store_true", help="Plot the resulting path.")
    args = parser.parse_args()

    initial_state, goal_state = get_initial_goal_states()
    start = np.array(initial_state[:2])
    goal = np.array(goal_state[:2])

    raw_obstacles = get_obstacles()
    obstacles = build_obstacle_list(raw_obstacles)
    bounds = compute_bounds(obstacles, start, goal)

    planner = PlanarRRTPlanner(
        bounds=bounds,
        obstacles=obstacles,
        step_size=args.step_size,
        max_iterations=args.max_iters,
        goal_sample_rate=args.goal_rate,
        clearance=args.clearance,
    )
    points = planner.plan(start, goal)
    summarize_path(points)
    states = convert_points_to_states(points)
    output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.output)
    save_to_json(states, output_path)
    print(f"Saved RRT path to {output_path}")

    if args.plot:
        plot_path(points, obstacles)


if __name__ == "__main__":
    main()
