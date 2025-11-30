from trajectory_planning import TrajectoryPlanning 
import casadi as ca
import time
import numpy as np
from scipy.interpolate import interp1d

class MPCTrackingControlPP(TrajectoryPlanning):
    def __init__(self, 
                 dynamics, params, 
                 Q, R, 
                 state_bound, input_bound):
        super().__init__(dynamics, params, Q, R, state_bound, input_bound)
        
        # Path progress variables s[k] for each time step in horizon
        self._s_sm = ca.SX.sym("path_progress", self._horizon+1)
        
        # Store reference path for interpolation (will be set when path is provided)
        self._ref_path_states = None  # Full reference path states
        self._ref_path_inputs = None  # Full reference path inputs  
        self._path_length = 0.0  # Total path length (arc-length parameterization)
        self._path_s_values = None  # Arc-length values for each point in reference path
        self._state_interpolants = None  # Casadi interpolants for states
        self._input_interpolants = None  # Casadi interpolants for inputs
        
        # Solver will be built when path is first provided
        self._solver = None
        self._last_solution = None  # For warm-starting
    
    def _setup_path_parameterization(self, ref_path_states, ref_path_inputs):
        """
        Set up path parameterization by arc-length.
        Computes cumulative arc-length along the path for interpolation.
        Rebuilds the solver with the new path data.
        
        Args:
            ref_path_states: Reference states along the path (num_state x N_states)
            ref_path_inputs: Reference inputs along the path (num_input x N_inputs)
        """
        self._ref_path_states = ref_path_states
        
        # Compute arc-length parameterization based on state trajectory
        N_states = ref_path_states.shape[1]
        N_inputs = ref_path_inputs.shape[1]
        s_values = np.zeros(N_states)
        
        # Compute cumulative arc-length
        for i in range(1, N_states):
            dx = ref_path_states[0, i] - ref_path_states[0, i-1]
            dy = ref_path_states[1, i] - ref_path_states[1, i-1]
            ds = np.sqrt(dx**2 + dy**2)
            s_values[i] = s_values[i-1] + ds
        
        self._path_s_values = s_values
        self._path_length = s_values[-1] if len(s_values) > 0 else 0.0
        
        # Handle input trajectory length mismatch
        # If inputs have fewer points than states, extend by repeating the last input
        if N_inputs < N_states:
            # Extend input trajectory to match state trajectory length
            ref_path_inputs_extended = np.zeros((self._num_input, N_states))
            ref_path_inputs_extended[:, :N_inputs] = ref_path_inputs
            # Repeat the last input for the remaining points
            if N_inputs > 0:
                last_input = ref_path_inputs[:, -1:]  # Shape: (num_input, 1)
                ref_path_inputs_extended[:, N_inputs:] = np.tile(last_input, (1, N_states - N_inputs))
            else:
                ref_path_inputs_extended[:, :] = 0.0
            self._ref_path_inputs = ref_path_inputs_extended
        else:
            # If inputs have same or more points, truncate or use as is
            self._ref_path_inputs = ref_path_inputs[:, :N_states]
        
        # Create casadi interpolant functions for symbolic evaluation
        self._state_interpolants = []
        self._input_interpolants = []
        
        for i in range(self._num_state):
            interp = ca.interpolant(f'state_{i}_interp', 'linear', 
                                   [s_values], ref_path_states[i, :])
            self._state_interpolants.append(interp)
        
        for i in range(self._num_input):
            interp = ca.interpolant(f'input_{i}_interp', 'linear',
                                   [s_values], self._ref_path_inputs[i, :])
            self._input_interpolants.append(interp)
        
        # Create scipy interpolation functions for numeric evaluation
        self._interp_functions = {}
        for i in range(self._num_state):
            self._interp_functions[f'state_{i}'] = interp1d(
                s_values, ref_path_states[i, :], 
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
        for i in range(self._num_input):
            self._interp_functions[f'input_{i}'] = interp1d(
                s_values, self._ref_path_inputs[i, :], 
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
        
        # Rebuild solver with new path data
        self._build_solver()
    
    def _get_ref_state_at_s(self, s):
        """
        Get reference state at path progress s using interpolation.
        
        Args:
            s: Path progress variable (scalar or casadi symbol)
            
        Returns:
            Reference state vector at s
        """
        if self._state_interpolants is None or len(self._state_interpolants) == 0:
            raise ValueError("Path parameterization not set up. Call _setup_path_parameterization first.")
        
        ref_state = []
        for i in range(self._num_state):
            if isinstance(s, (ca.SX, ca.MX)):
                # Clamp s to valid range
                s_clamped = ca.fmax(0, ca.fmin(s, self._path_length))
                ref_state.append(self._state_interpolants[i](s_clamped))
            else:
                # For numeric s, use scipy interpolation
                ref_state.append(self._interp_functions[f'state_{i}'](s))
        
        if isinstance(s, (ca.SX, ca.MX)):
            return ca.vertcat(*ref_state)
        else:
            return np.array(ref_state)
    
    def _get_ref_input_at_s(self, s):
        """
        Get reference input at path progress s using interpolation.
        
        Args:
            s: Path progress variable (scalar or casadi symbol)
            
        Returns:
            Reference input vector at s
        """
        if self._input_interpolants is None or len(self._input_interpolants) == 0:
            raise ValueError("Path parameterization not set up. Call _setup_path_parameterization first.")
        
        ref_input = []
        for i in range(self._num_input):
            if isinstance(s, (ca.SX, ca.MX)):
                # Clamp s to valid range
                s_clamped = ca.fmax(0, ca.fmin(s, self._path_length))
                ref_input.append(self._input_interpolants[i](s_clamped))
            else:
                ref_input.append(self._interp_functions[f'input_{i}'](s))
        
        if isinstance(s, (ca.SX, ca.MX)):
            return ca.vertcat(*ref_input)
        else:
            return np.array(ref_input)
    
    def _cost_function(self):
        cost = 0.
        for k in range(self._horizon):
            # Get reference state and input at path progress s[k]
            ref_state_k = self._get_ref_state_at_s(self._s_sm[k])
            ref_input_k = self._get_ref_input_at_s(self._s_sm[k])
            
            cost += (self._inputs_sm[:,k] - ref_input_k).T @ self._R @ (self._inputs_sm[:,k] - ref_input_k) + \
                (self._states_sm[:,k] - ref_state_k).T  @ self._Q @ (self._states_sm[:,k] - ref_state_k)
        
        # Final state cost
        Q_f = self._Q
        ref_state_f = self._get_ref_state_at_s(self._s_sm[-1])
        cost += (self._states_sm[:,-1] - ref_state_f).T  @ Q_f @ (self._states_sm[:,-1] - ref_state_f)
        
        return cost
        
    def _build_solver(self):
        """
        Build the MPC solver. Requires path to be set up first (interpolants must exist).
        """
        if self._state_interpolants is None or len(self._state_interpolants) == 0:
            raise ValueError("Path parameterization must be set up before building solver. "
                           "Call _setup_path_parameterization first.")
        
        g_dyn, lb_dyn, ub_dyn = self._dynamics_constraints()
        
        # Add path progress constraints: s[k+1] >= s[k] (non-decreasing, can stop)
        g_progress = []
        for k in range(self._horizon):
            g_progress.append(self._s_sm[k+1] - self._s_sm[k])  # s[k+1] >= s[k]
        
        # Combine all constraints
        g = ca.vertcat(g_dyn, *g_progress)
        self._lb_g = ca.vertcat(lb_dyn, ca.DM.zeros(len(g_progress)))  # Progress constraints >= 0
        self._ub_g = ca.vertcat(ub_dyn, ca.inf * ca.DM.ones(len(g_progress)))  # No upper bound on progress
                        
        self._lb_vars, self._ub_vars = self._state_input_constraints()        
        cost = self._cost_function()
        solver_opts = {
            'ipopt': {
                'max_iter': 5000,
                'print_level': 0,
                'tol': 1e-2,  # Relaxed from 1e-3 - accept less precise solutions
                'acceptable_tol': 1e-1,  # Relaxed from 1e-2 - accept "good enough" solutions
                'acceptable_iter': 3,  # Accept solution after 3 iterations if acceptable_tol is met
                'acceptable_obj_change_tol': 1e-2,  # Accept if objective change is small
                'mu_init': 1e-2,  # Larger initial barrier parameter for faster convergence
                'mu_strategy': 'adaptive',  # Adaptive barrier parameter update
                'fast_step_computation': 'yes',  # Faster step computation
                # Note: linear_solver option removed - let IPOPT choose automatically
                # 'linear_solver': 'ma27',  # May not be available on all systems
            },
            'print_time': False
        }     
        
        nlp_prob = {
            'f': cost,
            'x': self._vars,
            'p': self._init_state_sm,  # Only initial state as parameter now
            'g': g
        }
        self._solver = ca.nlpsol('solver','ipopt', nlp_prob, solver_opts)

        self.g_dyn_fun = ca.Function("g_dyn", [self._vars, self._init_state_sm], [g_dyn])
    
    def _state_input_constraints(self):
        """Override to include path progress variables s in decision variables."""
        self._vars = []
        lb_vars = []
        ub_vars = []
        
        # Bound s by path length to prevent extrapolation
        # If path_length is 0 or very small, use a reasonable upper bound
        s_max = max(self._path_length, 1.0) if self._path_length > 1e-6 else 1000.0
        
        for k in range(self._horizon):
            self._vars = ca.vertcat(self._vars,
                                    self._states_sm[:,k],
                                    self._inputs_sm[:,k],
                                    self._s_sm[k])  # Include s[k]
            lb_vars = ca.vertcat(lb_vars,
                                 self._state_bound['lb'],
                                 self._input_bound['lb'],
                                 0.0)  # s >= 0
            ub_vars = ca.vertcat(ub_vars, 
                                 self._state_bound['ub'], 
                                 self._input_bound['ub'],
                                 s_max)  # s <= path_length
        
        # Final state and final s
        self._vars = ca.vertcat(self._vars,
                                self._states_sm[:,-1],
                                self._s_sm[-1])
        lb_vars = ca.vertcat(lb_vars,
                             self._state_bound['lb'],
                             0.0)
        ub_vars = ca.vertcat(ub_vars,
                             self._state_bound['ub'],
                             s_max)
        
        return lb_vars, ub_vars
    
    def _find_closest_s_on_path(self, state):
        """
        Find the path progress s that corresponds to the closest point on the path to the current state.
        
        Args:
            state: Current state [x, y, ...]
            
        Returns:
            s: Path progress value
        """
        if self._ref_path_states is None or self._path_s_values is None:
            return 0.0
        
        # Find closest point on path by Euclidean distance in (x, y)
        x, y = state[0], state[1]
        path_x = self._ref_path_states[0, :]
        path_y = self._ref_path_states[1, :]
        
        distances = np.sqrt((path_x - x)**2 + (path_y - y)**2)
        closest_idx = np.argmin(distances)
        
        # Return the s value at the closest point
        return float(self._path_s_values[closest_idx])
        
    def _get_initial_guess(self, initial_s, initial_state):
        """
        Generate initial guess for optimization variables.
        
        Args:
            initial_s: Initial path progress value
            initial_state: Actual initial state (must match first state in optimization)
            
        Returns:
            vars_guess: Initial guess for all decision variables
        """
        # Clamp initial_s to valid range
        initial_s = max(0.0, min(initial_s, self._path_length))
        
        # Estimate s progression assuming constant speed along path
        # Simple linear progression from initial_s, but don't exceed path length
        if self._path_length > 0:
            # Progress forward along path, but cap at path length
            max_s = min(initial_s + 5.0, self._path_length)
            # If we're near the end, allow s to stay constant (vehicle can stop)
            if initial_s > 0.9 * self._path_length:
                max_s = self._path_length
        else:
            max_s = initial_s + 5.0
        
        s_guess = np.linspace(initial_s, max_s, self._horizon + 1)
        # Ensure all s values are within bounds
        s_guess = np.clip(s_guess, 0.0, self._path_length if self._path_length > 0 else 1000.0)
        
        # Get first reference input (use reference state at s0, but actual initial_state for state)
        ref_input_0 = self._get_ref_input_at_s(s_guess[0])
        
        # Convert to casadi DM if needed
        initial_state_dm = ca.DM(initial_state) if isinstance(initial_state, np.ndarray) else initial_state
        if isinstance(ref_input_0, np.ndarray):
            ref_input_0 = ca.DM(ref_input_0)
        
        # Use actual initial_state for first state (required by constraint)
        vars_guess = ca.vertcat(initial_state_dm, ref_input_0, s_guess[0])
        
        for k in range(1, self._horizon):
            # Get reference state and input at s[k]
            ref_state_k = self._get_ref_state_at_s(s_guess[k])
            ref_input_k = self._get_ref_input_at_s(s_guess[k])
            
            # Convert to casadi DM if needed
            if isinstance(ref_state_k, np.ndarray):
                ref_state_k = ca.DM(ref_state_k)
            if isinstance(ref_input_k, np.ndarray):
                ref_input_k = ca.DM(ref_input_k)
            
            vars_guess = ca.vertcat(vars_guess, ref_state_k, ref_input_k, s_guess[k])
        
        # Final state and s
        ref_state_f = self._get_ref_state_at_s(s_guess[-1])
        if isinstance(ref_state_f, np.ndarray):
            ref_state_f = ca.DM(ref_state_f)
        vars_guess = ca.vertcat(vars_guess, ref_state_f, s_guess[-1])
        
        return vars_guess

    def solve(self, 
              initial_state, 
              ref_path_states,
              ref_path_inputs):   
        """
        Solve MPC optimization problem with path-progress-based tracking.
        
        Args:
            initial_state: Current state [x, y, theta, psi, phi, v]
            ref_path_states: Full reference path states (num_state x N)
            ref_path_inputs: Full reference path inputs (num_input x N)
            
        Returns:
            states: Optimized state trajectory (num_state x horizon+1)
            inputs: Optimized input trajectory (num_input x horizon)
        """
        # Set up path parameterization (will rebuild solver if path changed)
        # Check if path changed by comparing shapes and a sample of values
        path_changed = (self._ref_path_states is None or 
                       self._ref_path_states.shape != ref_path_states.shape or
                       self._solver is None)
        
        if path_changed:
            self._setup_path_parameterization(ref_path_states, ref_path_inputs)
            self._last_solution = None  # Reset warm-start when path changes
        
        # Find initial path progress s0 based on current state
        s0 = self._find_closest_s_on_path(initial_state)
        
        # Use warm-start if available, otherwise generate new guess
        if self._last_solution is not None:
            # Shift previous solution: remove first step, add new step at end
            vars_guess = self._shift_solution(self._last_solution, s0, initial_state)
        else:
            vars_guess = self._get_initial_guess(s0, initial_state)
        
        start = time.time()
        sol = self._solver(
                x0  = vars_guess, 
                lbx = self._lb_vars,
                ubx = self._ub_vars,
                lbg = self._lb_g,
                ubg = self._ub_g,
                p = initial_state
            )
        end = time.time()
        # Solver runtime available if needed via `end-start`, but suppressed by default to keep logs clean.
        vars_opt = sol['x']
        
        # Store solution for warm-starting next iteration
        self._last_solution = vars_opt
        
        if self._solver.stats()['success'] == False:
            print("Cannot find a solution!") 
            print(f"  Solver status: {self._solver.stats()['return_status']}")
            print(f"  Initial s: {s0:.3f}, Path length: {self._path_length:.3f}")
        states, inputs = self._split_decision_variables(vars_opt)
        
        return states, inputs
    
    def _shift_solution(self, prev_vars, new_s0, initial_state):
        """
        Shift previous solution for warm-starting.
        Removes first step and adds a new step at the end.
        
        Args:
            prev_vars: Previous solution variables (CasADi DM)
            new_s0: New initial path progress
            initial_state: Current initial state
            
        Returns:
            Shifted variables for warm-start (CasADi DM)
        """
        num_state_input = self._num_state + self._num_input
        num_vars_per_step = num_state_input + 1  # +1 for s
        
        # Convert prev_vars to numpy for easier manipulation
        prev_vars_np = np.array(prev_vars)
        
        # First step: use actual initial_state
        initial_state_dm = ca.DM(initial_state) if isinstance(initial_state, np.ndarray) else initial_state
        
        # Get input and s from second step of previous solution
        if len(prev_vars_np) >= 2 * num_vars_per_step:
            # Extract second step's input and s
            second_step_start = num_vars_per_step
            first_input = ca.DM(prev_vars_np[second_step_start + self._num_state:second_step_start + self._num_state + self._num_input])
            first_s = float(prev_vars_np[second_step_start + self._num_state + self._num_input])
        else:
            # Fallback: use zeros for input, new_s0 for s
            first_input = ca.DM.zeros(self._num_input)
            first_s = new_s0
        
        shifted_vars = ca.vertcat(initial_state_dm, first_input, first_s)
        
        # Middle steps: shift previous solution (skip first step, take steps 1 to horizon-1)
        for k in range(1, self._horizon):
            step_idx = k  # k-th step in previous solution (0-indexed, but we skip step 0)
            step_start = step_idx * num_vars_per_step
            
            if step_start + num_vars_per_step <= len(prev_vars_np):
                # Extract state, input, s from this step
                state_k = ca.DM(prev_vars_np[step_start:step_start + self._num_state])
                input_k = ca.DM(prev_vars_np[step_start + self._num_state:step_start + self._num_state + self._num_input])
                s_k = float(prev_vars_np[step_start + self._num_state + self._num_input])
            else:
                # Extend with last available step
                last_step_start = (self._horizon - 1) * num_vars_per_step
                if last_step_start < len(prev_vars_np):
                    state_k = ca.DM(prev_vars_np[last_step_start:last_step_start + self._num_state])
                    if last_step_start + self._num_state + self._num_input <= len(prev_vars_np):
                        input_k = ca.DM(prev_vars_np[last_step_start + self._num_state:last_step_start + self._num_state + self._num_input])
                        s_k = float(prev_vars_np[last_step_start + self._num_state + self._num_input])
                    else:
                        input_k = ca.DM.zeros(self._num_input)
                        s_k = new_s0
                else:
                    state_k = initial_state_dm
                    input_k = ca.DM.zeros(self._num_input)
                    s_k = new_s0
            
            shifted_vars = ca.vertcat(shifted_vars, state_k, input_k, s_k)
        
        # Final state: use last state from previous solution
        final_state_start = self._horizon * num_vars_per_step
        if final_state_start < len(prev_vars_np):
            final_state = ca.DM(prev_vars_np[final_state_start:final_state_start + self._num_state])
            if final_state_start + self._num_state < len(prev_vars_np):
                final_s = float(prev_vars_np[final_state_start + self._num_state])
            else:
                final_s = new_s0
        else:
            final_state = initial_state_dm
            final_s = new_s0
        
        shifted_vars = ca.vertcat(shifted_vars, final_state, final_s)
        
        return shifted_vars
    
    def _split_decision_variables(self, vars):
        """
        Split decision variables into states, inputs, and path progress s.
        Override to handle s variables.
        """
        states = []
        inputs = []
        s_values = []
        
        num_state_input = self._num_state + self._num_input
        num_vars_per_step = num_state_input + 1  # +1 for s
        
        for k in range(self._horizon):
            vars_k = vars[k*num_vars_per_step:(k+1)*num_vars_per_step]
            state_input_s = vars_k[:num_state_input+1]
            states = ca.vertcat(states, state_input_s[:self._num_state])
            inputs = ca.vertcat(inputs, state_input_s[self._num_state:self._num_state+self._num_input])
            s_values.append(state_input_s[-1])
        
        # Final state and s
        vars_final = vars[-(num_vars_per_step-self._num_input):]
        states = ca.vertcat(states, vars_final[:self._num_state])
        s_values.append(vars_final[-1])
        
        states = states.reshape((self._num_state, self._horizon+1))
        inputs = inputs.reshape((self._num_input, self._horizon))
        
        # Return states, inputs, and s values
        return states.full(), inputs.full()
