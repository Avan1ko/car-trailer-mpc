from trajectory_planning import TrajectoryPlanning 
import casadi as ca
import numpy as np
import time

class MPCTrackingControl(TrajectoryPlanning):
    def __init__(self, 
                 dynamics, params, 
                 Q, R, 
                 state_bound, input_bound):
        super().__init__(dynamics, params, Q, R, state_bound, input_bound)
        
        self._ref_states_sm = ca.SX.sym("reference_states", self._num_state, self._horizon+1)
        self._ref_inputs_sm = ca.SX.sym("reference_inputs", self._num_input, self._horizon)        
        
        self._build_solver()
    
    def _cost_function(self):
        cost = 0.
        for k in range(self._horizon):
            cost += (self._inputs_sm[:,k] - self._ref_inputs_sm[:,k]).T @ self._R @ (self._inputs_sm[:,k] - self._ref_inputs_sm[:,k]) + \
                (self._states_sm[:,k] - self._ref_states_sm[:,k]).T  @ self._Q @ (self._states_sm[:,k] - self._ref_states_sm[:,k])
        Q_f = self._Q
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
            'ipopt': {'max_iter': 5000, 'print_level': 0},
            # 'ipopt': {'max_iter': 0, 'print_level': 5},
            'print_time': False
        }     
        # print(self._ref_inputs_sm[:,0])
        # exit()
        # print(self._ref_states_sm.reshape((-1, 1)))
        # print(self._ref_inputs_sm.reshape((-1, 1)))        
        # exit()  
        nlp_prob = {
            'f': cost,
            'x': self._vars,
            'p': ca.vertcat(self._ref_states_sm.reshape((-1, 1)), 
                            self._ref_inputs_sm.reshape((-1, 1)),
                            self._init_state_sm),
            'g': g
        }
        self._solver = ca.nlpsol('solver','ipopt', nlp_prob, solver_opts)

        self.g_dyn_fun = ca.Function("g_dyn", [self._vars, self._init_state_sm], [g_dyn])
        self.cost_fun = ca.Function("cost", [self._vars, self._ref_states_sm, self._ref_inputs_sm], [cost])
        
    def _get_initial_guess(self, reference_states, reference_inputs):
        vars_guess = []
        for k in range(self._horizon):
            state_guess = reference_states[:,k]
            input_guess = reference_inputs[:,k]
            vars_guess = ca.vertcat(vars_guess, state_guess, input_guess)
        vars_guess = ca.vertcat(vars_guess, reference_states[:,-1])
        return vars_guess

    def solve(self, 
              initial_state, 
              reference_states,
              reference_inputs):   
        reference_states_flat = reference_states.T.reshape((-1, 1))
        reference_inputs_flat = reference_inputs.T.reshape((-1, 1))
        
        vars_guess = self._get_initial_guess(reference_states, reference_inputs)
        start = time.time()
        sol = self._solver(
                x0  = vars_guess, 
                lbx = self._lb_vars,
                ubx = self._ub_vars,
                lbg = self._lb_g,
                ubg = self._ub_g,
                p = ca.vertcat(reference_states_flat, 
                               reference_inputs_flat,
                               initial_state)
            )
        end = time.time()
        # Solver runtime available if needed via `end-start`, but suppressed by default to keep logs clean.
        vars_opt = sol['x']
        
        # print(f"g_dyn: {self.g_dyn_fun(vars_guess, initial_state)}")
        
        # print(vars_guess[:20])
        # print(reference_inputs_flat)
        # print(reference_states)
        # print(reference_inputs)
        # print(vars_opt[:20])
        
        # print(f"g_dyn: {self.g_dyn_fun(vars_opt, initial_state)}")
        # print(f"cost: {self.cost_fun(vars_opt, reference_states, reference_inputs)}")
        # exit()
        
        if self._solver.stats()['success'] == False:
            print("Cannot find a solution!") 
        states, inputs = self._split_decision_variables(vars_opt)
        
        return states, inputs


class MPCTrackingControlLinearized(TrajectoryPlanning):
    """
    Linearized MPC controller that linearizes dynamics about each reference point.
    This is a drop-in replacement for MPCTrackingControl.
    """
    def __init__(self, 
                 dynamics, params, 
                 Q, R, 
                 state_bound, input_bound):
        super().__init__(dynamics, params, Q, R, state_bound, input_bound)
        
        self._ref_states_sm = ca.SX.sym("reference_states", self._num_state, self._horizon+1)
        self._ref_inputs_sm = ca.SX.sym("reference_inputs", self._num_input, self._horizon)
        
        # Create symbolic variables for linearization
        self._x_ref_sm = ca.SX.sym("x_ref", self._num_state, 1)
        self._u_ref_sm = ca.SX.sym("u_ref", self._num_input, 1)
        
        # Compute Jacobians symbolically
        f_val = self._dynamics.f(self._x_ref_sm, self._u_ref_sm)
        self._A_sm = ca.jacobian(f_val, self._x_ref_sm)
        self._B_sm = ca.jacobian(f_val, self._u_ref_sm)
        
        # Create function for computing linearization matrices
        self._linearize_fun = ca.Function('linearize', 
                                         [self._x_ref_sm, self._u_ref_sm],
                                         [self._A_sm, self._B_sm, f_val])
        
        # Expand the function to inline it (this helps avoid free variables)
        try:
            self._linearize_fun = self._linearize_fun.expand()
        except:
            pass  # If expand doesn't work, continue without it
        
        self._build_solver()
    
    def _cost_function(self):
        cost = 0.
        for k in range(self._horizon):
            cost += (self._inputs_sm[:,k] - self._ref_inputs_sm[:,k]).T @ self._R @ (self._inputs_sm[:,k] - self._ref_inputs_sm[:,k]) + \
                (self._states_sm[:,k] - self._ref_states_sm[:,k]).T  @ self._Q @ (self._states_sm[:,k] - self._ref_states_sm[:,k])
        Q_f = self._Q
        cost += (self._states_sm[:,-1] - self._ref_states_sm[:,-1]).T  @ Q_f @ (self._states_sm[:,-1] - self._ref_states_sm[:,-1])
        
        return cost
    
    def _dynamics_constraints(self):
        """
        Override to use linearized dynamics about reference trajectory points.
        For each step k, linearize about (x_ref_k, u_ref_k):
        x_{k+1} = x_k + dt * (A_k * (x_k - x_ref_k) + B_k * (u_k - u_ref_k) + f(x_ref_k, u_ref_k))
        where A_k = ∂f/∂x|_{x_ref_k, u_ref_k} and B_k = ∂f/∂u|_{x_ref_k, u_ref_k}
        
        Note: We use the linearization function to compute A_k, B_k, f_ref_k numerically
        at solve time, then rebuild the constraint structure.
        """
        g_dyn = [] 
        g_dyn = ca.vertcat(g_dyn, self._states_sm[:,0] - self._init_state_sm)
        
        dt = self._dynamics._params["dt"]
        
        # Create placeholder constraints using nonlinear dynamics
        # These will be replaced with linearized versions at solve time
        for k in range(self._horizon):
            g_dyn = ca.vertcat(g_dyn, self._states_sm[:,k+1] - self._dynamics.compute_next_state(self._states_sm[:, k], self._inputs_sm[:, k]))
        
        lb_dyn = ca.DM.zeros(g_dyn.shape)
        ub_dyn = ca.DM.zeros(g_dyn.shape)
        
        return g_dyn, lb_dyn, ub_dyn
    
    def _build_linearized_constraints(self, reference_states, reference_inputs):
        """
        Build linearized dynamics constraints using numerical linearization matrices.
        This is called at solve time to rebuild the constraint structure.
        """
        g_dyn = [] 
        g_dyn = ca.vertcat(g_dyn, self._states_sm[:,0] - self._init_state_sm)
        
        dt = self._dynamics._params["dt"]
        
        # Compute linearization matrices for each step
        for k in range(self._horizon):
            x_ref_k = reference_states[:, k]
            u_ref_k = reference_inputs[:, k]
            
            # Compute linearization matrices numerically
            A_k, B_k, f_ref_k = self._linearize_fun(x_ref_k, u_ref_k)
            
            # Convert to numpy arrays
            A_k = np.array(A_k)
            B_k = np.array(B_k)
            f_ref_k = np.array(f_ref_k).flatten()
            
            # Use reference parameters (symbolic) in the constraint
            x_ref_k_sym = self._ref_states_sm[:, k]
            u_ref_k_sym = self._ref_inputs_sm[:, k]
            
            # Convert numerical matrices to CasADi DM for use in constraint
            # Linearized dynamics: x_{k+1} = x_k + dt * (f_ref_k + A_k * (x_k - x_ref_k) + B_k * (u_k - u_ref_k))
            x_next_linearized = self._states_sm[:, k] + dt * (
                ca.DM(f_ref_k) + 
                ca.DM(A_k) @ (self._states_sm[:, k] - x_ref_k_sym) + 
                ca.DM(B_k) @ (self._inputs_sm[:, k] - u_ref_k_sym)
            )
            
            g_dyn = ca.vertcat(g_dyn, self._states_sm[:, k+1] - x_next_linearized)
        
        return g_dyn
        
    def _build_solver(self):
        g_dyn, lb_dyn, ub_dyn = self._dynamics_constraints()
        g = g_dyn
        self._lb_g = lb_dyn
        self._ub_g = ub_dyn
                        
        self._lb_vars, self._ub_vars = self._state_input_constraints()        
        cost = self._cost_function()
        solver_opts = {
            'ipopt': {'max_iter': 5000, 'print_level': 0},
            'print_time': False
        }     
        nlp_prob = {
            'f': cost,
            'x': self._vars,
            'p': ca.vertcat(self._ref_states_sm.reshape((-1, 1)), 
                            self._ref_inputs_sm.reshape((-1, 1)),
                            self._init_state_sm),
            'g': g
        }
        self._solver = ca.nlpsol('solver','ipopt', nlp_prob, solver_opts)

        # Note: g_dyn_fun and cost_fun are not used in the linearized version
        # since we rebuild constraints at solve time
        # self.g_dyn_fun = ca.Function("g_dyn", [self._vars, self._init_state_sm], [g_dyn])
        self.cost_fun = ca.Function("cost", [self._vars, self._ref_states_sm, self._ref_inputs_sm], [cost])
        
    def _get_initial_guess(self, reference_states, reference_inputs):
        vars_guess = []
        for k in range(self._horizon):
            state_guess = reference_states[:,k]
            input_guess = reference_inputs[:,k]
            vars_guess = ca.vertcat(vars_guess, state_guess, input_guess)
        vars_guess = ca.vertcat(vars_guess, reference_states[:,-1])
        return vars_guess

    def _build_qp_matrices(self, reference_states, reference_inputs, initial_state):
        """
        Build QP problem matrices: H (Hessian), g (gradient), A (constraint matrix), b (constraint RHS)
        """
        # Build linearized constraints symbolically
        g_dyn = self._build_linearized_constraints(reference_states, reference_inputs)
        
        reference_states_flat = reference_states.T.reshape((-1, 1))
        reference_inputs_flat = reference_inputs.T.reshape((-1, 1))
        
        # Extract constraint matrix A = dg_dyn/dvars (this is constant since constraints are linear)
        A_const = ca.jacobian(g_dyn, self._vars)
        
        # Create function to evaluate A with reference parameters
        A_fun = ca.Function('A_fun',
                           [self._ref_states_sm.reshape((-1, 1)),
                            self._ref_inputs_sm.reshape((-1, 1)),
                            self._init_state_sm],
                           [A_const])
        A_matrix = A_fun(reference_states_flat, reference_inputs_flat, initial_state)
        
        # Evaluate constraints at zero to get constant term (b = -g_dyn(0))
        # Since g_dyn = A*x - b = 0, we have b = -g_dyn(0)
        vars_zero = ca.DM.zeros(self._vars.shape)
        constraint_fun = ca.Function('constraint_fun', 
                                     [self._vars,
                                      self._ref_states_sm.reshape((-1, 1)),
                                      self._ref_inputs_sm.reshape((-1, 1)),
                                      self._init_state_sm],
                                     [g_dyn])
        b_const = -constraint_fun(vars_zero, reference_states_flat, reference_inputs_flat, initial_state)
        
        # For the cost function, extract Hessian H and gradient g
        cost = self._cost_function()
        
        # Extract Hessian (H = d^2 cost / d vars^2)
        H_matrix = ca.hessian(cost, self._vars)[0]
        H_fun = ca.Function('H_fun',
                           [self._ref_states_sm.reshape((-1, 1)),
                            self._ref_inputs_sm.reshape((-1, 1))],
                           [H_matrix])
        H = H_fun(reference_states_flat, reference_inputs_flat)
        
        # Extract gradient (g = d cost / d vars, evaluated at zero)
        grad_cost = ca.gradient(cost, self._vars)
        grad_fun = ca.Function('grad_fun',
                              [self._vars,
                               self._ref_states_sm.reshape((-1, 1)),
                               self._ref_inputs_sm.reshape((-1, 1))],
                              [grad_cost])
        g = grad_fun(vars_zero, reference_states_flat, reference_inputs_flat)
        
        return H, g, A_matrix, b_const
    
    def solve(self, 
              initial_state, 
              reference_states,
              reference_inputs):   
        # Build QP problem matrices
        H, g, A, b = self._build_qp_matrices(reference_states, reference_inputs, initial_state)
        
        # Convert to numpy arrays for QP solver
        H = np.array(H)
        g = np.array(g).flatten()
        A = np.array(A)
        b = np.array(b).flatten()
        lbx = np.array(self._lb_vars).flatten()
        ubx = np.array(self._ub_vars).flatten()
        
        # QP problem: minimize (1/2) * x^T * H * x + g^T * x
        # subject to: A * x = b, lbx <= x <= ubx
        
        # Build QP problem structure
        qp_prob = {
            'h': ca.DM(H),
            'g': ca.DM(g),
            'a': ca.DM(A),
            'lbx': ca.DM(lbx),
            'ubx': ca.DM(ubx),
            'lbg': ca.DM(b),  # Equality constraints: A*x = b, so lbg = ubg = b
            'ubg': ca.DM(b)
        }
        
        # Create QP solver
        solver_opts = {
            'print_time': False,
            'printLevel': 'none'
        }
        
        # Use qpOASES or OSQP as QP solver (qpOASES is default in CasADi)
        try:
            solver = ca.qpsol('qp_solver', 'qpoases', qp_prob, solver_opts)
        except:
            # Fallback to OSQP if qpOASES is not available
            try:
                solver = ca.qpsol('qp_solver', 'osqp', qp_prob, solver_opts)
            except:
                # Fallback to CLP if others are not available
                solver = ca.qpsol('qp_solver', 'clp', qp_prob, solver_opts)
        
        vars_guess = self._get_initial_guess(reference_states, reference_inputs)
        vars_guess = np.array(vars_guess).flatten()
        
        start = time.time()
        sol = solver(
            x0=ca.DM(vars_guess),
            lbx=ca.DM(lbx),
            ubx=ca.DM(ubx),
            lbg=ca.DM(b),
            ubg=ca.DM(b)
        )
        end = time.time()
        
        vars_opt = np.array(sol['x']).flatten()
        
        if solver.stats()['success'] == False:
            print("Cannot find a solution!") 
        
        # Reshape to match expected format and convert to DM
        vars_opt_dm = ca.DM(vars_opt.reshape((-1, 1)))
        states, inputs = self._split_decision_variables(vars_opt_dm)
        
        return states, inputs
