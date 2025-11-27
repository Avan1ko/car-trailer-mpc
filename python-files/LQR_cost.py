import casadi as ca
import numpy as np
from scipy.linalg import solve_discrete_are
from truck_trailer_model import TruckTrailerModel


def lqr_riccati(params, dynamics, Q, R, x_goal, u_goal): 
    n = dynamics.num_state
    m = dynamics.num_input
    dt = params['dt']

    # === 2. Create discrete dynamics wrapper and linearization function ===
    # Dynamics discrete map: x_next = x + f(x,u)*dt
    x_sym = ca.SX.sym('x', n)
    u_sym = ca.SX.sym('u', m)

    f_cont = dynamics.f(x_sym, u_sym)
    x_next_sym = x_sym + f_cont * dt  # forward Euler discretization of kinematics

    # Jacobians of discrete map
    A_sym = ca.jacobian(x_next_sym, x_sym)
    B_sym = ca.jacobian(x_next_sym, u_sym)

    # Function that returns numeric A, B matrices
    lin_fun = ca.Function('lin_discrete', [x_sym, u_sym], [A_sym, B_sym])

    A_k, B_k = lin_fun(ca.DM(x_goal), ca.DM(u_goal))
    A = np.array(A_k).astype(float)
    B = np.array(B_k).astype(float)

    A_k, B_k = lin_fun(ca.DM(x_goal), ca.DM(u_goal))
    P = solve_discrete_are(A, B, Q, R)
    P = 0.5 * (P + P.T)  # numerical symmetrization
    return P 

# === 5. Distance-to-goal metric using the LQR P matrix ===
def lqr_distance(x_current, x_goal, params, dynamics, Q, R, u_goal):
    P = lqr_riccati(params, dynamics, Q, R, x_goal, u_goal)
    """Computes quadratic LQR-based closeness score."""
    dx = x_current - x_goal
    return float(dx @ P @ dx)


