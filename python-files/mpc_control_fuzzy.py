from trajectory_planning import TrajectoryPlanning 
import casadi as ca
import numpy as np
import time

class MPCTrackingControlFuzzy(TrajectoryPlanning):
    def __init__(self, 
                 dynamics, params, 
                 Q, R, 
                 state_bound, input_bound):
        super().__init__(dynamics, params, Q, R, state_bound, input_bound)
        
        self._ref_states_sm = ca.SX.sym("reference_states", self._num_state, self._horizon+1)
        self._ref_inputs_sm = ca.SX.sym("reference_inputs", self._num_input, self._horizon)
        self._q_weights_sm = ca.SX.sym("q_weights", self._num_state)
        self._r_weights_sm = ca.SX.sym("r_weights", self._num_input)
        self._last_solution = None  # warm start between calls
        
        self._build_solver()
    
    def _cost_function(self):
        cost = 0.
        Q_w = ca.diag(self._q_weights_sm) @ self._Q @ ca.diag(self._q_weights_sm)
        R_w = ca.diag(self._r_weights_sm) @ self._R @ ca.diag(self._r_weights_sm)
        for k in range(self._horizon):
            cost += (self._inputs_sm[:,k] - self._ref_inputs_sm[:,k]).T @ R_w @ (self._inputs_sm[:,k] - self._ref_inputs_sm[:,k]) + \
                (self._states_sm[:,k] - self._ref_states_sm[:,k]).T  @ Q_w @ (self._states_sm[:,k] - self._ref_states_sm[:,k])
        Q_f = Q_w
        cost += (self._states_sm[:,-1] - self._ref_states_sm[:,-1]).T  @ Q_f @ (self._states_sm[:,-1] - self._ref_states_sm[:,-1])
        
        return cost
        
    def _build_solver(self):
        g_dyn, lb_dyn, ub_dyn = self._dynamics_constraints()
        g = g_dyn
        self._lb_g = lb_dyn
        self._ub_g = ub_dyn
                        
        self._lb_vars, self._ub_vars = self._state_input_constraints()        
        cost = self._cost_function()
        solver_opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'tol': 1e-3,
                'acceptable_tol': 1e-2,
                'acceptable_iter': 5,
            },
            'print_time': False
        }     
        nlp_prob = {
            'f': cost,
            'x': self._vars,
            'p': ca.vertcat(self._ref_states_sm.reshape((-1, 1)), 
                            self._ref_inputs_sm.reshape((-1, 1)),
                            self._q_weights_sm,
                            self._r_weights_sm,
                            self._init_state_sm),
            'g': g
        }
        self._solver = ca.nlpsol('solver','ipopt', nlp_prob, solver_opts)

    def _get_initial_guess(self, reference_states, reference_inputs):
        vars_guess = []
        for k in range(self._horizon):
            state_guess = reference_states[:,k]
            input_guess = reference_inputs[:,k]
            vars_guess = ca.vertcat(vars_guess, state_guess, input_guess)
        vars_guess = ca.vertcat(vars_guess, reference_states[:,-1])
        return vars_guess

    def _shift_solution(self, vars_opt):
        """Shift previous optimal trajectory forward one step for warm start."""
        num_state = self._num_state
        num_input = self._num_input
        h = self._horizon
        step = num_state + num_input

        shifted = []
        for k in range(h - 1):
            start = (k + 1) * step
            chunk = vars_opt[start:start + step]
            shifted = ca.vertcat(shifted, chunk)
        last_state = vars_opt[-step:-num_input]
        last_input = vars_opt[-num_input:]
        shifted = ca.vertcat(shifted, last_state, last_input)
        shifted = ca.vertcat(shifted, last_state)
        return shifted

    def _compute_fuzzy_weights(self, current_state, reference_states):
        """Lightweight fuzzy rules to scale Q/R based on hitch angle and reversing."""
        psi = float(current_state[3])
        v = float(current_state[5])
        ref_v = float(reference_states[5, 0]) if reference_states.size > 0 else 0.0

        hitch_soft = 0.35
        hitch_norm = min(abs(psi) / hitch_soft, 1.0)
        reversing = (ref_v < -0.1) or (v < -0.1)

        q_weights = np.ones(self._num_state)
        r_weights = np.ones(self._num_input)

        hitch_gain = 1.0 + 2.0 * hitch_norm
        steer_gain = 1.0 + 1.2 * hitch_norm
        steer_rate_gain = 1.0 + 1.5 * hitch_norm
        if reversing:
            hitch_gain *= 1.1
            steer_gain *= 1.1
            steer_rate_gain *= 1.2

        q_weights[2] = max(1.0, steer_gain)       # theta
        q_weights[3] = max(1.0, hitch_gain)       # psi (hitch)
        q_weights[4] = max(1.0, steer_gain)       # phi
        r_weights[1] = max(1.0, steer_rate_gain)  # steering rate input

        q_weights = np.clip(q_weights, 1.0, 3.5)
        r_weights = np.clip(r_weights, 1.0, 3.5)

        return ca.DM(q_weights), ca.DM(r_weights)

    def solve(self, 
              initial_state, 
              reference_states,
              reference_inputs):   
        reference_states_flat = reference_states.T.reshape((-1, 1))
        reference_inputs_flat = reference_inputs.T.reshape((-1, 1))
        
        if self._last_solution is not None:
            vars_guess = self._shift_solution(self._last_solution)
        else:
            vars_guess = self._get_initial_guess(reference_states, reference_inputs)
        q_weights, r_weights = self._compute_fuzzy_weights(initial_state, reference_states)
        sol = self._solver(
                x0  = vars_guess, 
                lbx = self._lb_vars,
                ubx = self._ub_vars,
                lbg = self._lb_g,
                ubg = self._ub_g,
                p = ca.vertcat(reference_states_flat, 
                               reference_inputs_flat,
                               q_weights,
                               r_weights,
                               initial_state)
            )
        if not self._solver.stats()['success']:
            q_fallback = ca.DM.ones(self._num_state)
            r_fallback = ca.DM.ones(self._num_input)
            sol = self._solver(
                x0=vars_guess,
                lbx=self._lb_vars,
                ubx=self._ub_vars,
                lbg=self._lb_g,
                ubg=self._ub_g,
                p=ca.vertcat(reference_states_flat,
                             reference_inputs_flat,
                             q_fallback,
                             r_fallback,
                             initial_state)
            )
        if not self._solver.stats()['success']:
            return None, None

        vars_opt = sol['x']
        self._last_solution = vars_opt
        states, inputs = self._split_decision_variables(vars_opt)
        
        return states, inputs
