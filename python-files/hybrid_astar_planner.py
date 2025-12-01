import argparse
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import casadi as ca

from get_initial_goal_states import get_initial_goal_states
from get_obstacles import get_obstacles
from truck_trailer_model import TruckTrailerModel


@dataclass(order=True)
class PrioritizedNode:
    priority: float
    node: "Node" = field(compare=False)


@dataclass
class Node:
    x: float
    y: float
    theta: float
    psi: float
    phi: float
    v: float
    g_cost: float
    parent: Optional["Node"]

    def key(self, pos_res: float, ang_res: float) -> Tuple[int, int, int, int]:
        return (
            int(round(self.x / pos_res)),
            int(round(self.y / pos_res)),
            int(round(self.theta / ang_res)),
            int(round(self.psi / ang_res)),
        )


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def simulate_primitive(model: TruckTrailerModel, node: Node, phi_cmd: float, v_cmd: float, steps: int, dt: float) -> Node:
    """Integrate forward with constant steering angle (phi_cmd) and velocity (v_cmd)."""
    x = np.array([node.x, node.y, node.theta, node.psi, node.phi, node.v], dtype=float)
    u = np.array([0.0, 0.0], dtype=float)  # a, omega
    x[4] = phi_cmd
    x[5] = v_cmd
    for _ in range(steps):
        f = model.f(x, u)
        f_np = np.array(ca.DM(f)).flatten()
        x = x + f_np * dt
        x[2] = wrap_angle(x[2])
        x[3] = wrap_angle(x[3])
    return Node(
        x=x[0],
        y=x[1],
        theta=x[2],
        psi=x[3],
        phi=x[4],
        v=x[5],
        g_cost=node.g_cost + steps * dt * abs(v_cmd),
        parent=node,
    )


def vehicle_bounds(params, state: Node) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return (veh_center, trailer_center) as (x,y)."""
    model = TruckTrailerModel(params)
    xv, yv = model.get_vehicle_center(state.x, state.y, state.theta)
    xt, yt = model.get_trailer_center(state.x, state.y, state.theta, state.psi)
    return (float(xv), float(yv)), (float(xt), float(yt))


def collides(state: Node, obstacles, params, clearance: float) -> bool:
    (vx, vy), (tx, ty) = vehicle_bounds(params, state)
    rv = max(params["L1"], params["W1"]) * 0.5 + clearance
    rt = max(params["L2"], params["W2"]) * 0.5 + clearance
    for obs in obstacles:
        cx, cy = obs["center"]
        w = obs["width"] * 0.5
        h = obs["height"] * 0.5
        if cx - w - rv <= vx <= cx + w + rv and cy - h - rv <= vy <= cy + h + rv:
            return True
        if cx - w - rt <= tx <= cx + w + rt and cy - h - rt <= ty <= cy + h + rt:
            return True
    return False


def heuristic(node: Node, goal: Node) -> float:
    pos = math.hypot(goal.x - node.x, goal.y - node.y)
    ang = abs(wrap_angle(goal.theta - node.theta)) + abs(wrap_angle(goal.psi - node.psi))
    return pos + ang


def reconstruct(node: Node) -> List[Node]:
    path: List[Node] = []
    cur = node
    while cur:
        path.append(cur)
        cur = cur.parent
    path.reverse()
    return path


def hybrid_astar(
    start: Node,
    goal: Node,
    model: TruckTrailerModel,
    obstacles,
    params,
    steering_set,
    v_set,
    step_dt: float,
    steps_per_primitive: int,
    pos_res: float,
    ang_res: float,
    max_iter: int,
    goal_tol_xy: float,
    goal_tol_angle: float,
    clearance: float,
) -> Optional[List[Node]]:
    open_set: List[PrioritizedNode] = []
    heapq.heappush(open_set, PrioritizedNode(priority=heuristic(start, goal), node=start))
    visited = {}

    for _ in range(max_iter):
        if not open_set:
            return None
        current = heapq.heappop(open_set).node
        if (
            math.hypot(current.x - goal.x, current.y - goal.y) <= goal_tol_xy
            and abs(wrap_angle(current.theta - goal.theta)) <= goal_tol_angle
            and abs(wrap_angle(current.psi - goal.psi)) <= goal_tol_angle
        ):
            return reconstruct(current)

        key = current.key(pos_res, ang_res)
        if key in visited and visited[key] <= current.g_cost:
            continue
        visited[key] = current.g_cost

        for phi_cmd in steering_set:
            for v_cmd in v_set:
                nxt = simulate_primitive(model, current, phi_cmd, v_cmd, steps_per_primitive, step_dt)
                if collides(nxt, obstacles, params, clearance):
                    continue
                cost = nxt.g_cost + heuristic(nxt, goal)
                heapq.heappush(open_set, PrioritizedNode(priority=cost, node=nxt))
    return None


def write_initialize(path_nodes: List[Node], output: Path):
    positions = []
    headings = []
    hitches = []
    for n in path_nodes:
        positions.append([n.x, n.y])
        headings.append(n.theta - math.pi / 2.0)  # undo +pi/2 downstream
        hitches.append(n.psi)
    data = {"Positions": positions, "Headings": headings, "HitchAngles": hitches}
    output.write_text(json.dumps(data, indent=2))
    print(f"Wrote Hybrid A* path -> {output} (waypoints={len(positions)})")


def main():
    parser = argparse.ArgumentParser(description="Simple Python Hybrid A* for truck-trailer.")
    parser.add_argument("--output", type=str, default="initialize.json", help="Where to write the path JSON.")
    parser.add_argument("--step-dt", type=float, default=0.2, help="Integration dt for motion primitives.")
    parser.add_argument("--primitive-steps", type=int, default=5, help="Integration steps per primitive.")
    parser.add_argument("--pos-res", type=float, default=0.5, help="Grid resolution for visited states (m).")
    parser.add_argument("--ang-res", type=float, default=0.25, help="Grid resolution for visited states (rad).")
    parser.add_argument("--max-iter", type=int, default=40000, help="Max node expansions.")
    parser.add_argument("--goal-tol-xy", type=float, default=1.0, help="Goal position tolerance (m).")
    parser.add_argument("--goal-tol-angle", type=float, default=0.3, help="Goal angle tolerance (rad).")
    parser.add_argument("--clearance", type=float, default=0.5, help="Collision clearance (m).")
    parser.add_argument("--steer", type=float, nargs="+", default=[-0.4, 0.0, 0.4], help="Steering angles (rad).")
    parser.add_argument("--speed", type=float, nargs="+", default=[-2.0], help="Constant speeds for primitives (m/s). Use negative for reversing.")
    args = parser.parse_args()

    # Vehicle params (same as animation defaults)
    params = {"M": 0.15, "L1": 7.05, "L2": 12.45, "W1": 3.05, "W2": 2.95, "dt": args.step_dt, "horizon": 200}
    model = TruckTrailerModel(params)

    initial_state, goal_state = get_initial_goal_states()
    # initial_state/goal_state are [x,y,theta,psi]; add phi=0, v=0
    start_node = Node(
        x=initial_state[0],
        y=initial_state[1],
        theta=initial_state[2],
        psi=initial_state[3],
        phi=0.0,
        v=0.0,
        g_cost=0.0,
        parent=None,
    )
    goal_node = Node(
        x=goal_state[0],
        y=goal_state[1],
        theta=goal_state[2],
        psi=goal_state[3],
        phi=0.0,
        v=0.0,
        g_cost=0.0,
        parent=None,
    )

    obstacles = get_obstacles()

    path = hybrid_astar(
        start=start_node,
        goal=goal_node,
        model=model,
        obstacles=obstacles,
        params=params,
        steering_set=args.steer,
        v_set=args.speed,
        step_dt=args.step_dt,
        steps_per_primitive=args.primitive_steps,
        pos_res=args.pos_res,
        ang_res=args.ang_res,
        max_iter=args.max_iter,
        goal_tol_xy=args.goal_tol_xy,
        goal_tol_angle=args.goal_tol_angle,
        clearance=args.clearance,
    )

    if path is None:
        print("Hybrid A* failed to find a path.")
        return

    output_path = Path(args.output)
    write_initialize(path, output_path)


if __name__ == "__main__":
    main()
