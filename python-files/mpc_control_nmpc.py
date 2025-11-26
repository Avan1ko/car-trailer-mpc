from trajectory_planning import TrajectoryPlanning
import casadi as ca


class TruckTrailerNMPC(TrajectoryPlanning):
    """
    Nonlinear MPC using the full truck-trailer kinematic model.
    Reuses TrajectoryPlanning for constraints and decision variable layout.
    """

    def __init__(self, dynamics, params, Q, R, state_bound, input_bound):
        super().__init__(dynamics, params, Q, R, state_bound, input_bound)
        self._ref_states_sm = ca.SX.sym("reference_states", self._num_state, self._horizon + 1)
        self._ref_inputs_sm = ca.SX.sym("reference_inputs", self._num_input, self._horizon)
        self._last_solution = None  # warm start between calls
        self._build_solver()

    def _cost_function(self):
        cost = 0.0
        for k in range(self._horizon):
            dx = self._states_sm[:, k] - self._ref_states_sm[:, k]
            du = self._inputs_sm[:, k] - self._ref_inputs_sm[:, k]
            cost += dx.T @ self._Q @ dx + du.T @ self._R @ du
        dx_f = self._states_sm[:, -1] - self._ref_states_sm[:, -1]
        cost += dx_f.T @ self._Q @ dx_f
        return cost

    def _build_solver(self):
        g_dyn, lb_dyn, ub_dyn = self._dynamics_constraints()
        self._lb_g = lb_dyn
        self._ub_g = ub_dyn

        self._lb_vars, self._ub_vars = self._state_input_constraints()
        cost = self._cost_function()

        solver_opts = {
            "ipopt": {
                "max_iter": 2000,
                "print_level": 0,
                "tol": 1e-3,
                "acceptable_tol": 1e-2,
                "acceptable_iter": 5,
            },
            "print_time": False,
        }

        nlp_prob = {
            "f": cost,
            "x": self._vars,
            "p": ca.vertcat(
                self._ref_states_sm.reshape((-1, 1)),
                self._ref_inputs_sm.reshape((-1, 1)),
                self._init_state_sm,
            ),
            "g": g_dyn,
        }

        self._solver = ca.nlpsol("solver", "ipopt", nlp_prob, solver_opts)

    def _get_initial_guess(self, reference_states, reference_inputs):
        vars_guess = []
        for k in range(self._horizon):
            state_guess = reference_states[:, k]
            input_guess = reference_inputs[:, k]
            vars_guess = ca.vertcat(vars_guess, state_guess, input_guess)
        vars_guess = ca.vertcat(vars_guess, reference_states[:, -1])
        return vars_guess

    def _shift_solution(self, vars_opt):
        """Shift previous optimal trajectory forward one step for warm start."""
        num_state = self._num_state
        num_input = self._num_input
        h = self._horizon
        step = num_state + num_input

        shifted = []
        # shift states/inputs forward
        for k in range(h - 1):
            start = (k + 1) * step
            chunk = vars_opt[start:start + step]
            shifted = ca.vertcat(shifted, chunk)
        # last stage: reuse last state and last input
        last_state = vars_opt[-step:-num_input]
        last_input = vars_opt[-num_input:]
        shifted = ca.vertcat(shifted, last_state, last_input)
        # final state duplicated
        shifted = ca.vertcat(shifted, last_state)
        return shifted

    def solve(self, initial_state, reference_states, reference_inputs):
        reference_states_flat = reference_states.T.reshape((-1, 1))
        reference_inputs_flat = reference_inputs.T.reshape((-1, 1))
        if self._last_solution is not None:
            vars_guess = self._shift_solution(self._last_solution)
        else:
            vars_guess = self._get_initial_guess(reference_states, reference_inputs)

        sol = self._solver(
            x0=vars_guess,
            lbx=self._lb_vars,
            ubx=self._ub_vars,
            lbg=self._lb_g,
            ubg=self._ub_g,
            p=ca.vertcat(reference_states_flat, reference_inputs_flat, initial_state),
        )

        if not self._solver.stats()["success"]:
            return None, None

        vars_opt = sol["x"]
        self._last_solution = vars_opt
        states, inputs = self._split_decision_variables(vars_opt)
        return states, inputs
