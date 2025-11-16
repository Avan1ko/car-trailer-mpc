import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_rrt_path(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_obstacles(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rect_from_corners(obstacle: dict):
    xs = [obstacle["FL"]["X"], obstacle["FR"]["X"], obstacle["BL"]["X"], obstacle["BR"]["X"]]
    ys = [obstacle["FL"]["Y"], obstacle["FR"]["Y"], obstacle["BL"]["Y"], obstacle["BR"]["Y"]]
    center_x = sum(xs) / 4.0
    center_y = sum(ys) / 4.0
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    return center_x - width / 2.0, center_y - height / 2.0, width, height


def plot_rrt(path_file: str, obstacles_file: str):
    data = load_rrt_path(path_file)
    obstacles = load_obstacles(obstacles_file)

    xs = [pos[0] for pos in data["Positions"]]
    ys = [pos[1] for pos in data["Positions"]]

    fig, ax = plt.subplots()
    for obstacle in obstacles:
        x, y, width, height = rect_from_corners(obstacle)
        rect = Rectangle((x, y), width, height, facecolor="gray", alpha=0.4)
        ax.add_patch(rect)

    ax.plot(xs, ys, "-o", label="RRT path")
    ax.scatter(xs[0], ys[0], c="green", s=120, marker="s", label="Start")
    ax.scatter(xs[-1], ys[-1], c="red", s=120, marker="*", label="Goal")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    ax.set_title("Planar RRT path")

    plt.show()


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    rrt_path_file = os.path.join(script_dir, "rrt_path.json")
    obstacles_file = os.path.join(script_dir, "..", "obstacles.json")

    plot_rrt(rrt_path_file, obstacles_file)


if __name__ == "__main__":
    main()
