from trajectory_planning import TrajectoryPlanning 
import casadi as ca
import time
import numpy as np
from get_obstacles import get_obstacles


class MPCTrackingControlObs(TrajectoryPlanning):
    def __init__(self, 
                 dynamics, params, 
                 Q, R, 
                 state_bound, input_bound, obstacle_list=None):
        super().__init__(dynamics, params, Q, R, state_bound, input_bound)
        
        self._ref_states_sm = ca.SX.sym("reference_states", self._num_state, self._horizon+1)
        self._ref_inputs_sm = ca.SX.sym("reference_inputs", self._num_input, self._horizon)        

        self.obstacle_list = obstacle_list if obstacle_list is not None else []
        if len(self.obstacle_list) > 0:
            num_obstacles = len(self.obstacle_list)
            self._mus_sm = ca.SX.sym("mus", 8*num_obstacles, self._horizon+1)
            self._lams_sm = ca.SX.sym("lams", 8*num_obstacles, self._horizon+1)
        else:
            self._mus_sm = None
            self._lams_sm = None
        
        self._last_solution = None  # Add this to store previous solution
        
        # Build solver with or without obstacle constraints based on obstacle_list
        self._build_solver()
    
    def _cost_function(self):
        cost = 0.
        for k in range(self._horizon):
            cost += (self._inputs_sm[:,k] - self._ref_inputs_sm[:,k]).T @ self._R @ (self._inputs_sm[:,k] - self._ref_inputs_sm[:,k]) + \
                (self._states_sm[:,k] - self._ref_states_sm[:,k]).T  @ self._Q @ (self._states_sm[:,k] - self._ref_states_sm[:,k])
        Q_f = self._Q
        cost += (self._states_sm[:,-1] - self._ref_states_sm[:,-1]).T  @ Q_f @ (self._states_sm[:,-1] - self._ref_states_sm[:,-1])
        
        return cost
    
    def _obstacle_Hrep(self, obstacle):
        width = obstacle["width"]
        height = obstacle["height"]
        center = ca.DM([
            [obstacle["center"][0]],
            [obstacle["center"][1]],
        ])
        A = ca.DM([
            [ 1.,  0.],
            [ 0.,  1.],
            [-1.,  0.],
            [ 0., -1.],
        ])
        b = ca.DM([
            [width/2.],
            [height/2.],
            [width/2.],
            [height/2.],
        ])
        b += A @ center # Shift center of half-space representation 
                
        return A, b # Returns A matrix (4 by 2) and b vector (4 by 1) this is the obstacle's half-space representation  

    def _collision_constraints(self):
        # Rectangles
        d_min = 0.2 # Why does the plot get so weird when we make d_min high, is that supposed to happen?
        Gv, gv = self._dynamics.get_vehicle_Hrep() # Half-space representation of the vehicle (still centered at 0)
        Gt, gt = self._dynamics.get_trailer_Hrep() # Half-space representation of the trailer (still centered at 0)
        g_col = [] # Empty array for collision constraints
        lb_col = [] #lb for g_col
        ub_col = [] #ub for g_col
        for k in range(0, self._horizon+1): # Loop over entire prediction horizon
            x_rear = self._states_sm[0,k] # Get the x position of rear axle from the states vector
            y_rear = self._states_sm[1,k] # Get the y position of rear axle from the states vector   
            heading = self._states_sm[2,k]# Get the heading of the vehicle from the states vector
            hitch_angle = self._states_sm[3,k] # Get the hitch angle from the states vector 
            xv_center, yv_center = self._dynamics.get_vehicle_center(x_rear, y_rear, heading) # Compute the x,y position of the center of the vehicle
            pv_center = ca.vertcat(xv_center, yv_center) # Make a vector out of this x and y component of the center
            
            # Do the same for pt_center, the center point of the trailer
            xt_center, yt_center = self._dynamics.get_trailer_center(x_rear, y_rear, heading, hitch_angle)
            pt_center = ca.vertcat(xt_center, yt_center)
            # Making rotation matrices for vehicle and trailer
            Rv = ca.SX.zeros((2, 2))
            Rv[0,0] =  ca.cos(heading)
            Rv[0,1] = -ca.sin(heading)
            Rv[1,0] =  ca.sin(heading)
            Rv[1,1] =  ca.cos(heading)
            
            Rt = ca.SX.zeros((2, 2))
            Rt[0,0] =  ca.cos(heading+hitch_angle)
            Rt[0,1] = -ca.sin(heading+hitch_angle)
            Rt[1,0] =  ca.sin(heading+hitch_angle)
            Rt[1,1] =  ca.cos(heading+hitch_angle)
            
            for i, obstacle in enumerate(self.obstacle_list):
                Ao, bo = self._obstacle_Hrep(obstacle) # Getting the obstacle half-space representation of current obstacle
                # constraint 1 for Truck: here we use mu and [0,1,2,3] and [8,9,10,11]
                g_col = ca.vertcat(g_col, gv.T @ self._mus_sm[i*8:i*8+4,k] 
                                          - (Ao @ pv_center - bo).T @ self._lams_sm[i*8:i*8+4,k] 
                                          + d_min)
                # Do it again for trailer: here we use mu and lam [4,5,6,7] and [12,13,14,15]
                g_col = ca.vertcat(g_col, gt.T @ self._mus_sm[i*8+4:i*8+8,k] 
                                          - (Ao @ pt_center - bo).T @ self._lams_sm[i*8+4:i*8+8,k] 
                                          + d_min)
                # Now we added 2 constraints
                lb_col = ca.vertcat(lb_col, -ca.inf*ca.DM.ones(1))
                lb_col = ca.vertcat(lb_col, -ca.inf*ca.DM.ones(1))
                ub_col = ca.vertcat(ub_col,  ca.DM.zeros(1))
                ub_col = ca.vertcat(ub_col,  ca.DM.zeros(1))

                # constraint equation 2 for Truck:
                g_col = ca.vertcat(g_col, Gv.T @ self._mus_sm[i*8:i*8+4,k] + Rv.T @ Ao.T @ self._lams_sm[i*8:i*8+4,k]) 
                
                # Do it again for trailer: 
                g_col = ca.vertcat(g_col, Gt.T @ self._mus_sm[i*8+4:i*8+8,k] + Rt.T @ Ao.T @ self._lams_sm[i*8+4:i*8+8,k])
                
                # Now we should add 4 lb and ub, because we added 2x2 lines of constraints
                lb_col = ca.vertcat(lb_col, -1e-5*ca.DM.ones(2)) #was 1e-5
                lb_col = ca.vertcat(lb_col, -1e-5*ca.DM.ones(2))
                ub_col = ca.vertcat(ub_col,  1e-5*ca.DM.ones(2))
                ub_col = ca.vertcat(ub_col,  1e-5*ca.DM.ones(2))
                
                
                # Constraint 3 for Truck:
                g_col = ca.vertcat(g_col, ca.norm_2(Ao.T @ self._lams_sm[i*8:i*8+4,k]) - 1)
                
                # Do it again for trailer:
                g_col = ca.vertcat(g_col, ca.norm_2(Ao.T @ self._lams_sm[i*8+4:i*8+8,k]) - 1)
                
                # Nowe we should add 2 lb and ub, because we added 2x1 lines of constraints
                lb_col = ca.vertcat(lb_col, -ca.inf*ca.DM.ones(1))
                lb_col = ca.vertcat(lb_col, -ca.inf*ca.DM.ones(1))
                ub_col = ca.vertcat(ub_col,  ca.DM.zeros(1))
                ub_col = ca.vertcat(ub_col,  ca.DM.zeros(1))
        
        return g_col, lb_col, ub_col
    
    def _state_input_constraints(self):
        self._vars = []
        lb_vars = []
        ub_vars = []
        num_obstacles = len(self.obstacle_list)
        for k in range(self._horizon):
            self._vars = ca.vertcat(self._vars,
                                    self._states_sm[:,k],
                                    self._inputs_sm[:,k],
                                    self._mus_sm[:,k],
                                    self._lams_sm[:,k])
            lb_vars = ca.vertcat(lb_vars,
                                 self._state_bound['lb'],
                                 self._input_bound['lb'],
                                 ca.DM.zeros(8*num_obstacles),
                                 ca.DM.zeros(8*num_obstacles))
            ub_vars = ca.vertcat(ub_vars, 
                                 self._state_bound['ub'], 
                                 self._input_bound['ub'],
                                 ca.inf*ca.DM.ones(8*num_obstacles),
                                 ca.inf*ca.DM.ones(8*num_obstacles))
            
        # Constraint for final state (controls not needed)
        self._vars = ca.vertcat(self._vars,
                                self._states_sm[:,-1],
                                self._mus_sm[:,-1],
                                self._lams_sm[:,-1])
        lb_vars = ca.vertcat(lb_vars,
                             self._state_bound['lb'],
                             ca.DM.zeros(8*num_obstacles),
                             ca.DM.zeros(8*num_obstacles))
        ub_vars = ca.vertcat(ub_vars,
                             self._state_bound['ub'],
                             ca.inf*ca.DM.ones(8*num_obstacles),
                             ca.inf*ca.DM.ones(8*num_obstacles))
        
        return lb_vars, ub_vars


    def _build_solver(self):
        g_dyn, lb_dyn, ub_dyn = self._dynamics_constraints()
        g = g_dyn
        self._lb_g = lb_dyn
        self._ub_g = ub_dyn

        if self.obstacle_list and len(self.obstacle_list) > 0:
            g_col, lb_col, ub_col = self._collision_constraints()
            g = ca.vertcat(g, g_col)
            self._lb_g = ca.vertcat(self._lb_g, lb_col)
            self._ub_g = ca.vertcat(self._ub_g, ub_col)
                        
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
        num_obstacles = len(self.obstacle_list) if self.obstacle_list else 0

        for k in range(self._horizon):
            state_guess = reference_states[:,k]
            input_guess = reference_inputs[:,k]
            vars_guess = ca.vertcat(vars_guess, state_guess, input_guess)

            if num_obstacles > 0:
                vars_guess = ca.vertcat(vars_guess,
                                        100.*ca.DM.ones(8*num_obstacles),
                                        ca.kron(ca.DM.ones(num_obstacles), 
                                                ca.DM([100, 105, 110, 115, 100, 105, 110, 115])))

        vars_guess = ca.vertcat(vars_guess, reference_states[:,-1])

        if num_obstacles > 0:
            vars_guess = ca.vertcat(vars_guess,
                                    100.*ca.DM.ones(8*num_obstacles),
                                    ca.kron(ca.DM.ones(num_obstacles), 
                                            ca.DM([100, 105, 110, 115, 100, 105, 110, 115])))
        
        return vars_guess
        
    def _split_decision_variables(self, vars):
        """Override parent method to handle mus and lams when obstacles are present."""
        states = []
        inputs = []
        
        num_obstacles = len(self.obstacle_list) if self.obstacle_list else 0
        num_state_input = self._num_state + self._num_input
        
        if num_obstacles > 0:
            # With obstacles: each step has states, inputs, mus (8*num_obs), lams (8*num_obs)
            num_vars_per_step = num_state_input + 8*num_obstacles + 8*num_obstacles
            
            for k in range(self._horizon):
                start_idx = k * num_vars_per_step
                state_k = vars[start_idx:start_idx + self._num_state]
                input_k = vars[start_idx + self._num_state:start_idx + num_state_input]
                states = ca.vertcat(states, state_k)
                inputs = ca.vertcat(inputs, input_k)
            
            # Final state (no inputs, just states, mus, lams)
            final_start = self._horizon * num_vars_per_step
            final_state = vars[final_start:final_start + self._num_state]
            states = ca.vertcat(states, final_state)
        else:
            # Without obstacles: use parent class logic
            num_vars_per_step = num_state_input
            
            for k in range(self._horizon):
                vars_k = vars[k*num_vars_per_step:(k+1)*num_vars_per_step]
                states = ca.vertcat(states, vars_k[:self._num_state])
                inputs = ca.vertcat(inputs, vars_k[self._num_state:])
            
            # Final state
            vars_final = vars[-(num_vars_per_step - self._num_input):]
            states = ca.vertcat(states, vars_final[:self._num_state])
        
        states = states.reshape((self._num_state, self._horizon+1))
        inputs = inputs.reshape((self._num_input, self._horizon))
        
        return states.full(), inputs.full()

    def solve(self, 
              initial_state, 
              reference_states,
              reference_inputs):   
        reference_states_flat = reference_states.T.reshape((-1, 1))
        reference_inputs_flat = reference_inputs.T.reshape((-1, 1))

        # Use appropriate initial guess based on whether obstacles exist
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
        self._last_solution = vars_opt
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
