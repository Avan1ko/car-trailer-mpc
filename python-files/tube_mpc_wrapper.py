"""
Tube MPC wrapper for robust truck-trailer NMPC.

Preserves nominal model.
Adds linear-feedback tube for disturbance robustness.

This module implements:
- Linearization of nominal dynamics
- Discrete LQR feedback gain computation
- Tube bounds computation for constraint tightening
- TubeMPCWrapper class that combines nominal MPC with feedback control
"""

import casadi as ca
import numpy as np
from scipy.linalg import solve_discrete_are, solve_continuous_are
from scipy.linalg import expm
from mpc_control import MPCTrackingControl


def linearize_dynamics(f, x_nom, u_nom):
    """
    Linearize dynamics function f(x, u) about nominal point (x_nom, u_nom).
    
    Args:
        f: CasADi function f(x, u) that returns state derivative
        x_nom: Nominal state (numpy array or CasADi DM)
        u_nom: Nominal input (numpy array or CasADi DM)
        
    Returns:
        A: State Jacobian matrix (numpy array, n x n)
        B: Input Jacobian matrix (numpy array, n x m)
    """
    # Convert to CasADi SX for symbolic differentiation
    n = len(x_nom)
    m = len(u_nom)
    
    x_sym = ca.SX.sym("x", n, 1)
    u_sym = ca.SX.sym("u", m, 1)
    
    # Evaluate dynamics symbolically
    f_sym = f(x_sym, u_sym)
    
    # Compute Jacobians
    A_sym = ca.jacobian(f_sym, x_sym)
    B_sym = ca.jacobian(f_sym, u_sym)
    
    # Create function to evaluate at given point
    linearize_fun = ca.Function('linearize', [x_sym, u_sym], [A_sym, B_sym])
    
    # Evaluate at nominal point
    A, B = linearize_fun(x_nom, u_nom)
    
    # Convert to numpy
    A = np.array(A)
    B = np.array(B)
    
    return A, B


def discretize_dynamics(A_c, B_c, dt):
    """
    Discretize continuous-time dynamics using zero-order hold.
    
    For continuous-time system: x_dot = A_c * x + B_c * u
    Returns discrete-time: x_{k+1} = A_d * x_k + B_d * u_k
    
    Args:
        A_c: Continuous-time state matrix (n x n)
        B_c: Continuous-time input matrix (n x m)
        dt: Time step
        
    Returns:
        A_d: Discrete-time state matrix (n x n)
        B_d: Discrete-time input matrix (n x m)
    """
    n = A_c.shape[0]
    m = B_c.shape[1]
    
    # Build augmented matrix for matrix exponential
    M = np.block([[A_c, B_c], 
                  [np.zeros((m, n + m))]])
    
    # Compute matrix exponential
    M_exp = expm(M * dt)
    
    # Extract discrete-time matrices
    A_d = M_exp[:n, :n]
    B_d = M_exp[:n, n:]
    
    return A_d, B_d


def compute_feedback_gain(A_c, B_c, Q_lqr, R_lqr, dt):
    """
    Compute discrete-time LQR feedback gain K.
    
    Solves discrete-time Riccati equation:
    P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A
    K = (R + B^T P B)^{-1} B^T P A
    
    Args:
        A_c: Continuous-time state matrix (n x n) from linearization
        B_c: Continuous-time input matrix (n x m) from linearization
        Q_lqr: State cost matrix (n x n)
        R_lqr: Input cost matrix (m x m)
        dt: Time step for discretization
        
    Returns:
        K: Feedback gain matrix (m x n)
    """
    # Discretize continuous-time dynamics
    A_d, B_d = discretize_dynamics(A_c, B_c, dt)
    
    # Solve discrete-time Riccati equation
    try:
        P = solve_discrete_are(A_d, B_d, Q_lqr, R_lqr)
    except Exception as e:
        # Fallback: use simple pole placement or identity
        print(f"Warning: Could not solve Riccati equation ({e}), using simple gain")
        # Use simple proportional gain as fallback
        # K = alpha * B_d^T, where alpha is small
        alpha = 0.1
        K = alpha * B_d.T
        return K
    
    # Compute feedback gain
    try:
        K = np.linalg.solve(R_lqr + B_d.T @ P @ B_d, B_d.T @ P @ A_d)
    except:
        # Fallback if solve fails
        K = np.linalg.pinv(R_lqr + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
    
    return K


def compute_tube_bounds(A_c, B_c, K, disturbance_bound, dt, method='box'):
    """
    Compute tightened constraint bounds for Tube MPC.
    
    Computes an invariant set Z such that if x_nom - x_meas ∈ Z,
    then the closed-loop system (A + BK) keeps the error bounded.
    
    Args:
        A_c: Continuous-time state matrix (n x n) from linearization
        B_c: Continuous-time input matrix (n x m) from linearization
        K: Feedback gain (m x n)
        disturbance_bound: Maximum disturbance magnitude (scalar or vector)
        dt: Time step
        method: 'box' for simple box bounds, 'ellipsoid' for ellipsoidal (not implemented)
        
    Returns:
        state_tube: Dictionary with 'lb' and 'ub' for tightened state bounds
        input_tube: Dictionary with 'lb' and 'ub' for tightened input bounds
    """
    n = A_c.shape[0]
    m = B_c.shape[1]
    
    # Discretize continuous-time dynamics
    A_d, B_d = discretize_dynamics(A_c, B_c, dt)
    
    # Closed-loop matrix: e_{k+1} = (A_d + B_d * K) * e_k + w_k
    A_cl = A_d + B_d @ K
    
    # Check stability
    eigvals = np.linalg.eigvals(A_cl)
    max_eig = np.max(np.abs(eigvals))
    
    if max_eig >= 1.0:
        print(f"Warning: Closed-loop system may be unstable (max |eig| = {max_eig:.3f})")
        # Use conservative bounds
        rho = 0.99  # Slightly less than 1
    else:
        rho = max_eig
    
    # Convert disturbance bound to vector if scalar
    if np.isscalar(disturbance_bound):
        w_max = np.ones(n) * disturbance_bound
    else:
        w_max = np.array(disturbance_bound)
        if len(w_max) != n:
            w_max = np.ones(n) * np.max(w_max)
    
    # Compute invariant set using simple box approximation
    # For system e_{k+1} = A_cl e_k + w_k, we want to find Z such that
    # if e_k ∈ Z, then e_{k+1} ∈ Z for all w_k ∈ W
    # Using box bounds: Z = {e : |e_i| <= z_i}
    
    # Simple approximation: iterate to find fixed point
    # z = max(|A_cl| @ z + w_max) where |A_cl| is element-wise absolute value
    z = np.ones(n) * 0.1  # Initial guess
    max_iter = 100
    tol = 1e-6
    
    for i in range(max_iter):
        z_new = np.abs(A_cl) @ z + w_max
        if np.allclose(z, z_new, atol=tol):
            break
        z = z_new
    
    # Add safety margin (10%)
    z = z * 1.1
    
    # Compute input tube bounds from state tube
    # u_fb = K * e, so |u_fb| <= |K| @ z
    u_tube = np.abs(K) @ z
    
    # Create tightened bounds
    # These will be subtracted from original bounds
    state_tube = {
        'lb': -z,
        'ub': z
    }
    
    input_tube = {
        'lb': -u_tube,
        'ub': u_tube
    }
    
    return state_tube, input_tube


class TubeMPCWrapper:
    """
    Wrapper class for Tube MPC that combines nominal MPC with feedback control.
    
    The wrapper:
    1. Runs nominal MPC to get u_nom
    2. Linearizes dynamics about nominal trajectory
    3. Computes feedback gain K
    4. Applies feedback: u_applied = u_nom + K * (x_meas - x_nom)
    5. Uses tightened constraints in MPC
    """
    
    def __init__(self, 
                 dynamics, params,
                 Q, R,
                 state_bound, input_bound,
                 Q_lqr=None, R_lqr=None,
                 disturbance_bound=0.02):
        """
        Initialize Tube MPC wrapper.
        
        Args:
            dynamics: Nominal dynamics model (TruckTrailerModel)
            params: System parameters dict
            Q: State cost matrix for MPC
            R: Input cost matrix for MPC
            state_bound: Dict with 'lb' and 'ub' for state constraints
            input_bound: Dict with 'lb' and 'ub' for input constraints
            Q_lqr: State cost matrix for LQR (defaults to Q)
            R_lqr: Input cost matrix for LQR (defaults to R)
            disturbance_bound: Maximum disturbance magnitude
        """
        self._dynamics = dynamics
        self._params = params
        self._disturbance_bound = disturbance_bound
        
        # Use Q, R for LQR if not specified
        if Q_lqr is None:
            Q_lqr = Q
        if R_lqr is None:
            R_lqr = R
        
        self._Q_lqr = Q_lqr
        self._R_lqr = R_lqr
        
        # Compute initial tube bounds (will be updated each iteration)
        # Start with zero tightening
        n = dynamics.num_state
        m = dynamics.num_input
        
        # Create tightened bounds (initially same as original)
        self._state_bound_tight = {
            'lb': state_bound['lb'],
            'ub': state_bound['ub']
        }
        self._input_bound_tight = {
            'lb': input_bound['lb'],
            'ub': input_bound['ub']
        }
        
        # Create nominal MPC with tightened constraints
        self._nominal_mpc = MPCTrackingControl(
            dynamics, params,
            Q, R,
            self._state_bound_tight,
            self._input_bound_tight
        )
        
        # Initialize feedback gain (will be computed each iteration)
        self._K = np.zeros((m, n))
        self._x_nom_current = None
        self._u_nom_current = None
        
    def _update_tube_bounds(self, x_nom, u_nom):
        """
        Update tube bounds based on current nominal trajectory.
        
        Args:
            x_nom: Nominal state trajectory (n x horizon+1)
            u_nom: Nominal input trajectory (m x horizon)
        """
        # Linearize about first point of trajectory
        x_nom_0 = x_nom[:, 0]
        u_nom_0 = u_nom[:, 0]
        
        # Get dynamics function (returns continuous-time derivative)
        def f_dyn(x, u):
            return self._dynamics.f(x, u)
        
        # Linearize (returns continuous-time A_c, B_c)
        A_c, B_c = linearize_dynamics(f_dyn, x_nom_0, u_nom_0)
        
        # Compute feedback gain (handles discretization internally)
        dt = self._params['dt']
        K = compute_feedback_gain(A_c, B_c, self._Q_lqr, self._R_lqr, dt)
        self._K = K
        
        # Compute tube bounds (handles discretization internally)
        state_tube, input_tube = compute_tube_bounds(
            A_c, B_c, K, self._disturbance_bound, dt
        )
        
        # Tighten constraints: X_tight = X_original ⊖ Z
        # For box constraints: [lb_tight, ub_tight] = [lb_orig + z_lb, ub_orig - z_ub]
        state_lb_orig = np.array(self._nominal_mpc._state_bound['lb']).flatten()
        state_ub_orig = np.array(self._nominal_mpc._state_bound['ub']).flatten()
        
        input_lb_orig = np.array(self._nominal_mpc._input_bound['lb']).flatten()
        input_ub_orig = np.array(self._nominal_mpc._input_bound['ub']).flatten()
        
        # Tighten: subtract tube from upper bounds, add to lower bounds
        # X_tight = X_original ⊖ Z (Minkowski difference)
        state_lb_tight = state_lb_orig + state_tube['lb']
        state_ub_tight = state_ub_orig - state_tube['ub']
        
        input_lb_tight = input_lb_orig + input_tube['lb']
        input_ub_tight = input_ub_orig - input_tube['ub']
        
        # Ensure bounds remain valid (lb <= ub)
        # If tightening makes bounds invalid, use original bounds
        invalid_state = np.any(state_lb_tight > state_ub_tight)
        invalid_input = np.any(input_lb_tight > input_ub_tight)
        
        if invalid_state or invalid_input:
            print("Warning: Tube bounds too large, using original bounds")
            state_lb_tight = state_lb_orig
            state_ub_tight = state_ub_orig
            input_lb_tight = input_lb_orig
            input_ub_tight = input_ub_orig
        
        # Update bounds (convert back to CasADi DM)
        self._state_bound_tight = {
            'lb': ca.DM(state_lb_tight),
            'ub': ca.DM(state_ub_tight)
        }
        self._input_bound_tight = {
            'lb': ca.DM(input_lb_tight),
            'ub': ca.DM(input_ub_tight)
        }
        
        # Update MPC bounds (will be used in next solve)
        # Note: We don't rebuild the solver here to avoid overhead
        # The bounds are used when constructing the constraint vectors
        self._nominal_mpc._state_bound = self._state_bound_tight
        self._nominal_mpc._input_bound = self._input_bound_tight
    
    def solve(self, x_meas, reference_states, reference_inputs):
        """
        Solve Tube MPC and return applied control.
        
        Args:
            x_meas: Measured/estimated current state (numpy array, n x 1)
            reference_states: Reference state trajectory (n x horizon+1)
            reference_inputs: Reference input trajectory (m x horizon)
            
        Returns:
            u_applied: Applied control input (numpy array, m x 1)
            x_nom: Nominal state trajectory (n x horizon+1)
            u_nom: Nominal input trajectory (m x horizon)
        """
        # Rebuild solver with current tightened bounds if needed
        # Check if bounds have changed significantly
        if not hasattr(self, '_bounds_initialized'):
            # First time: rebuild solver with tightened bounds
            self._nominal_mpc._state_bound = self._state_bound_tight
            self._nominal_mpc._input_bound = self._input_bound_tight
            self._nominal_mpc._build_solver()
            self._bounds_initialized = True
        
        # Solve nominal MPC with tightened constraints
        # Use x_meas as initial condition for nominal trajectory
        states_nom, inputs_nom = self._nominal_mpc.solve(
            x_meas, reference_states, reference_inputs
        )
        
        # Store current nominal trajectory
        self._x_nom_current = states_nom
        self._u_nom_current = inputs_nom
        
        # Update tube bounds based on nominal trajectory
        # This updates K and prepares bounds for next iteration
        self._update_tube_bounds(states_nom, inputs_nom)
        
        # Rebuild solver with new tightened bounds for next iteration
        self._nominal_mpc._state_bound = self._state_bound_tight
        self._nominal_mpc._input_bound = self._input_bound_tight
        self._nominal_mpc._build_solver()
        
        # Compute feedback control
        x_nom_0 = states_nom[:, 0]  # Current nominal state
        e = x_meas - x_nom_0  # Error
        u_fb = self._K @ e  # Feedback control
        
        # Nominal control (first step)
        u_nom_0 = inputs_nom[:, 0]
        
        # Applied control
        u_applied = u_nom_0 + u_fb
        
        return u_applied, states_nom, inputs_nom
    
    def get_feedback_gain(self):
        """Get current feedback gain matrix K."""
        return self._K.copy()
    
    def get_nominal_trajectory(self):
        """Get current nominal state and input trajectories."""
        return self._x_nom_current, self._u_nom_current

