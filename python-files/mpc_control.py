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
        Placeholder dynamics constraints - will be replaced with linearized version at solve time.
        """
        g_dyn = [] 
        g_dyn = ca.vertcat(g_dyn, self._states_sm[:,0] - self._init_state_sm)
        
        for k in range(self._horizon):
            g_dyn = ca.vertcat(g_dyn, self._states_sm[:,k+1] - self._dynamics.compute_next_state(self._states_sm[:, k], self._inputs_sm[:, k]))
        
        lb_dyn = ca.DM.zeros(g_dyn.shape)
        ub_dyn = ca.DM.zeros(g_dyn.shape)
        
        return g_dyn, lb_dyn, ub_dyn
        
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
        """
        Solve the linearized MPC problem as a QP using CasADi's qpsol.
        
        The linearized dynamics are:
        x_{k+1} = x_k + dt * (f_ref_k + A_k * (x_k - x_ref_k) + B_k * (u_k - u_ref_k))
        """
        dt = self._dynamics._params["dt"]
        
        # Get initial guess
        vars_guess = self._get_initial_guess(reference_states, reference_inputs)
        vars_guess = np.array(vars_guess).flatten()
        
        # Build constraint matrix A and RHS vector b for linearized dynamics
        # Constraint: A_eq @ vars = b_eq
        n_vars = len(vars_guess)
        n_constraints = self._num_state * (self._horizon + 1)  # Initial condition + dynamics
        
        A_eq = np.zeros((n_constraints, n_vars))
        b_eq = np.zeros(n_constraints)
        
        # Initial condition constraint: x_0 = initial_state
        idx_constraint = 0
        idx_var = 0
        for i in range(self._num_state):
            A_eq[idx_constraint, idx_var + i] = 1.0
            b_eq[idx_constraint] = initial_state[i]
            idx_constraint += 1
        
        # Dynamics constraints
        for k in range(self._horizon):
            # Get reference trajectory point
            x_ref_k = reference_states[:, k]
            u_ref_k = reference_inputs[:, k]
            
            # Compute linearization matrices
            A_k, B_k, f_ref_k = self._linearize_fun(x_ref_k, u_ref_k)
            A_k = np.array(A_k)
            B_k = np.array(B_k)
            f_ref_k = np.array(f_ref_k).flatten()
            
            # Variable indices
            idx_x_k = k * (self._num_state + self._num_input)
            idx_u_k = idx_x_k + self._num_state
            idx_x_kp1 = (k + 1) * (self._num_state + self._num_input)
            
            # Constraint: x_{k+1} = x_k + dt * (f_ref_k + A_k * (x_k - x_ref_k) + B_k * (u_k - u_ref_k))
            # Rearranged: x_{k+1} - (I + dt*A_k)*x_k - dt*B_k*u_k = dt*(f_ref_k - A_k*x_ref_k - B_k*u_ref_k)
            
            for i in range(self._num_state):
                # Coefficient for x_{k+1}
                A_eq[idx_constraint, idx_x_kp1 + i] = 1.0
                
                # Coefficient for x_k: -(I + dt*A_k)
                for j in range(self._num_state):
                    if i == j:
                        A_eq[idx_constraint, idx_x_k + j] = -1.0 - dt * A_k[i, j]
                    else:
                        A_eq[idx_constraint, idx_x_k + j] = -dt * A_k[i, j]
                
                # Coefficient for u_k: -dt*B_k
                for j in range(self._num_input):
                    A_eq[idx_constraint, idx_u_k + j] = -dt * B_k[i, j]
                
                # RHS
                b_eq[idx_constraint] = dt * (f_ref_k[i] - A_k[i, :] @ x_ref_k - B_k[i, :] @ u_ref_k)
                
                idx_constraint += 1
        
        # Build Hessian matrix H for quadratic cost
        # Cost: 0.5 * x^T @ H @ x + g^T @ x
        H = np.zeros((n_vars, n_vars))
        g = np.zeros(n_vars)
        
        for k in range(self._horizon):
            idx_x_k = k * (self._num_state + self._num_input)
            idx_u_k = idx_x_k + self._num_state
            
            # State cost: (x_k - x_ref_k)^T @ Q @ (x_k - x_ref_k)
            # Expands to: x_k^T @ Q @ x_k - 2 * x_ref_k^T @ Q @ x_k + constant
            Q_array = np.array(self._Q)
            for i in range(self._num_state):
                for j in range(self._num_state):
                    H[idx_x_k + i, idx_x_k + j] += Q_array[i, j]
                g[idx_x_k + i] += -2.0 * Q_array[i, :] @ reference_states[:, k]
            
            # Input cost: (u_k - u_ref_k)^T @ R @ (u_k - u_ref_k)
            R_array = np.array(self._R)
            for i in range(self._num_input):
                for j in range(self._num_input):
                    H[idx_u_k + i, idx_u_k + j] += R_array[i, j]
                g[idx_u_k + i] += -2.0 * R_array[i, :] @ reference_inputs[:, k]
        
        # Terminal cost: (x_N - x_ref_N)^T @ Q_f @ (x_N - x_ref_N)
        idx_x_N = self._horizon * (self._num_state + self._num_input)
        Q_f = np.array(self._Q)
        for i in range(self._num_state):
            for j in range(self._num_state):
                H[idx_x_N + i, idx_x_N + j] += Q_f[i, j]
            g[idx_x_N + i] += -2.0 * Q_f[i, :] @ reference_states[:, -1]
        
        # Make H symmetric (in case of numerical errors)
        H = 0.5 * (H + H.T)
        
        # Ensure H is positive definite by adding small regularization if needed
        min_eig = np.min(np.linalg.eigvals(H))
        if min_eig < 1e-8:
            H += (1e-8 - min_eig + 1e-6) * np.eye(n_vars)
        
        # Convert to CasADi sparse format
        H_sparse = ca.DM(H)
        g_sparse = ca.DM(g)
        A_sparse = ca.DM(A_eq)
        b_sparse = ca.DM(b_eq)
        
        # Bounds
        lb = np.array(self._lb_vars).flatten()
        ub = np.array(self._ub_vars).flatten()
        lbx = ca.DM(lb)
        ubx = ca.DM(ub)
        
        # Try solving with QP solver
        try:
            # Create QP problem
            qp = {
                'h': H_sparse.sparsity(),
                'a': A_sparse.sparsity()
            }
            
            # Try different QP solvers
            solver = None
            solver_name = None
            
            for qp_solver in ['qpoases', 'osqp', 'qrqp']:
                try:
                    solver = ca.conic('qp_solver', qp_solver, qp, {'error_on_fail': False})
                    solver_name = qp_solver
                    break
                except:
                    continue
            
            if solver is None:
                raise Exception("No QP solver available")
            
            # Solve QP
            start = time.time()
            sol = solver(
                h=H_sparse,
                g=g_sparse,
                a=A_sparse,
                lba=b_sparse,
                uba=b_sparse,
                lbx=lbx,
                ubx=ubx,
                x0=ca.DM(vars_guess)
            )
            end = time.time()
            
            # Check if solution is valid
            if solver.stats()['return_status'] not in ['SUCCESS', 'SOLVED']:
                raise Exception(f"QP solver failed with status: {solver.stats()['return_status']}")
            
            vars_opt = np.array(sol['x']).flatten()
            
        except Exception as e:
            # Fall back to NLP solver with linearized constraints
            print(f"QP solve failed ({e}), using NLP solver with linearized dynamics")
            
            # Build linearized NLP problem on-the-fly
            g_dyn_list = [self._states_sm[:,0] - self._init_state_sm]
            
            for k in range(self._horizon):
                x_ref_k = reference_states[:, k]
                u_ref_k = reference_inputs[:, k]
                
                A_k, B_k, f_ref_k = self._linearize_fun(x_ref_k, u_ref_k)
                
                # Linearized dynamics as symbolic expression
                x_next_lin = self._states_sm[:, k] + dt * (
                    ca.DM(f_ref_k) + 
                    ca.DM(A_k) @ (self._states_sm[:, k] - ca.DM(x_ref_k)) + 
                    ca.DM(B_k) @ (self._inputs_sm[:, k] - ca.DM(u_ref_k))
                )
                
                g_dyn_list.append(self._states_sm[:, k+1] - x_next_lin)
            
            g_dyn = ca.vertcat(*g_dyn_list)
            
            # Build NLP with linearized dynamics
            reference_states_flat = reference_states.T.reshape((-1, 1))
            reference_inputs_flat = reference_inputs.T.reshape((-1, 1))
            
            cost = self._cost_function()
            
            nlp_prob = {
                'f': cost,
                'x': self._vars,
                'p': ca.vertcat(self._ref_states_sm.reshape((-1, 1)), 
                                self._ref_inputs_sm.reshape((-1, 1)),
                                self._init_state_sm),
                'g': g_dyn
            }
            
            solver_opts = {
                'ipopt': {'max_iter': 2000, 'print_level': 0, 'tol': 1e-6},
                'print_time': False
            }
            nlp_solver = ca.nlpsol('nlp_solver','ipopt', nlp_prob, solver_opts)
            
            start = time.time()
            sol = nlp_solver(
                x0=ca.DM(vars_guess),
                lbx=self._lb_vars,
                ubx=self._ub_vars,
                lbg=ca.DM.zeros(g_dyn.shape),
                ubg=ca.DM.zeros(g_dyn.shape),
                p=ca.vertcat(reference_states_flat, 
                             reference_inputs_flat,
                             initial_state)
            )
            end = time.time()
            
            vars_opt = np.array(sol['x']).flatten()
            
            if nlp_solver.stats()['success'] == False:
                print("Cannot find a solution!")
        
        # Reshape to match expected format and convert to DM
        vars_opt_dm = ca.DM(vars_opt.reshape((-1, 1)))
        states, inputs = self._split_decision_variables(vars_opt_dm)
        
        return states, inputs

import casadi as ca
import numpy as np
import time

class MPCTrackingControlLTV(TrajectoryPlanning):

    def __init__(self, dynamics, params, Q, R, state_bound, input_bound):
        super().__init__(dynamics, params, Q, R, state_bound, input_bound)

        self.N = params["horizon"]
        self.nx = dynamics.num_state
        self.nu = dynamics.num_input

        # Reference symbols
        self.Xref = ca.SX.sym("Xref", self.nx, self.N+1)
        self.Uref = ca.SX.sym("Uref", self.nu, self.N)

        # Build decision vector
        self.lbx, self.ubx = self._state_input_constraints()  # fills self._vars

        # Linearized constraints (symbolic)
        self.g_lin = self._build_linearized_constraints_symbolic()

        # Cost
        self.cost = self._build_cost_symbolic()

        # Hessian, gradient, constraint Jacobian
        self.H_sym = ca.hessian(self.cost, self._vars)[0]
        self.grad_sym = ca.gradient(self.cost, self._vars)
        self.Aeq_sym = ca.jacobian(self.g_lin, self._vars)

        self.H_fun = ca.Function("H_fun",[self.Xref, self.Uref],[self.H_sym])
        self.grad_fun = ca.Function("grad_fun",[self.Xref,self.Uref,self._init_state_sm,self._vars],[self.grad_sym])
        self.Aeq_fun = ca.Function("Aeq_fun",[self.Xref,self.Uref,self._init_state_sm],[self.Aeq_sym])
        self.g_fun = ca.Function("g_fun",[self.Xref,self.Uref,self._init_state_sm,self._vars],[self.g_lin])


    # --------------------- SYMBOLIC LINEARIZED DYNAMICS ---------------------
    def _build_linearized_constraints_symbolic(self):
        g = []
        g = ca.vertcat(g, self._states_sm[:,0] - self._init_state_sm)

        x_sym = ca.SX.sym("x", self.nx)
        u_sym = ca.SX.sym("u", self.nu)
        f_disc = self._dynamics.compute_next_state(x_sym, u_sym)

        A_sym = ca.jacobian(f_disc, x_sym)
        B_sym = ca.jacobian(f_disc, u_sym)
        Afun = ca.Function("Afun",[x_sym,u_sym],[A_sym])
        Bfun = ca.Function("Bfun",[x_sym,u_sym],[B_sym])
        Ffun = ca.Function("Ffun",[x_sym,u_sym],[f_disc])

        for k in range(self.N):
            xk = self._states_sm[:, k]
            uk = self._inputs_sm[:, k]
            xkp1 = self._states_sm[:, k+1]
            xr = self.Xref[:, k]
            ur = self.Uref[:, k]

            f_ref = Ffun(xr, ur)
            A_k = Afun(xr, ur)
            B_k = Bfun(xr, ur)

            x_lin = f_ref + A_k @ (xk - xr) + B_k @ (uk - ur)
            g = ca.vertcat(g, xkp1 - x_lin)

        return g


    # -------------------------- COST FUNCTION ------------------------------
    def _build_cost_symbolic(self):
        J = 0
        for k in range(self.N):
            dx = self._states_sm[:,k] - self.Xref[:,k]
            du = self._inputs_sm[:,k] - self.Uref[:,k]
            J += dx.T @ self._Q @ dx + du.T @ self._R @ du

        dxN = self._states_sm[:,-1] - self.Xref[:,-1]
        J += dxN.T @ self._Q @ dxN
        return J


    # ---------------------------- QP SOLVE ---------------------------------
    def solve(self, x0, Xref_val, Uref_val):

        Xref_dm = ca.DM(Xref_val)
        Uref_dm = ca.DM(Uref_val)
        x0_dm = ca.DM(x0).reshape((-1,1))

        # Build QP matrices
        H = self.H_fun(Xref_dm, Uref_dm)
        Aeq = self.Aeq_fun(Xref_dm, Uref_dm, x0_dm)
        z0 = ca.DM.zeros(self._vars.shape)
        g0 = self.grad_fun(Xref_dm, Uref_dm, x0_dm, z0)
        g_dyn_0 = self.g_fun(Xref_dm, Uref_dm, x0_dm, z0)
        beq = -g_dyn_0

        # ------------------- LOW LEVEL QP (conic/qpoases) -------------------

        # Sparsity patterns
        qp_struct = {
            "h": H.sparsity(),
            "a": Aeq.sparsity()
        }

        solver = ca.conic("solver","qpoases", qp_struct)

        sol = solver(
            h=H,
            g=g0,
            a=Aeq,
            lba=beq,
            uba=beq,
            lbx=self.lbx,
            ubx=self.ubx,
        )

        z = sol["x"]
        X, U = self._split_decision_variables(z)
        return X, U

