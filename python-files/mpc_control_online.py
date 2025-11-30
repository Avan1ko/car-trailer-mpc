"""
Online Adaptive MPC Controller with Disturbance Learning

This module implements an MPC controller that learns disturbance parameters online
and adapts constraints accordingly. Supports three learning methods:
- 'rls': Recursive Least Squares (exponential forgetting)
- 'moving_average': Moving average of parameter estimates
- 'simple_estimation': Simple heuristic-based estimation

The controller learns:
- friction_coeff: Friction coefficient affecting acceleration
- slippage_coeff: Slippage coefficient affecting steering
- process_noise_std: Process noise standard deviation
- lateral_slip_gain: Lateral slip gain affecting sideways drift

Usage:
    from mpc_control_online import MPCTrackingControlOnline
    
    controller = MPCTrackingControlOnline(
        dynamics=model,
        params=params,
        Q=Q, R=R,
        state_bound=state_bound,
        input_bound=input_bound,
        learning_method='rls',  # or 'moving_average' or 'simple_estimation'
        window_size=50,
        initial_disturbance_params=None,
        adaptive_constraint_tightening=True
    )
    
    # In simulation loop:
    states, inputs = controller.solve(
        initial_state=current_state,
        reference_states=ref_states,
        reference_inputs=ref_inputs,
        actual_state=previous_actual_state,  # For learning
        update_estimator=True
    )
    
    # Get learned parameters:
    learned_params = controller.get_disturbance_params()
"""

from trajectory_planning import TrajectoryPlanning 
import casadi as ca
import numpy as np
import time
from collections import deque

class OnlineDisturbanceEstimator:
    """
    Lightweight online disturbance parameter estimator with multiple learning methods
    """
    def __init__(self, method='moving_average', window_size=50, initial_params=None):
        """
        Args:
            method: 'rls', 'moving_average', or 'simple_estimation'
            window_size: Fixed window size for history
            initial_params: Initial guess for disturbance parameters
        """
        self.method = method
        self.window_size = window_size
        
        # Initialize parameters with defaults
        if initial_params is None:
            self.params = {
                'friction_coeff': 1.0,
                'slippage_coeff': 1.0,
                'process_noise_std': 0.0,
                'lateral_slip_gain': 0.0
            }
        else:
            self.params = initial_params.copy()
        
        # History storage (fixed window)
        self.prediction_errors = deque(maxlen=window_size)  # List of state prediction errors
        self.control_inputs = deque(maxlen=window_size)    # List of control inputs
        self.states = deque(maxlen=window_size)            # List of actual states
        self.predicted_states = deque(maxlen=window_size)  # List of predicted states
        
        # RLS-specific parameters
        if method == 'rls':
            # Initialize RLS for each parameter
            self.rls_params = {
                'friction_coeff': {'theta': 1.0, 'P': 100.0, 'lambda': 0.99},
                'slippage_coeff': {'theta': 1.0, 'P': 100.0, 'lambda': 0.99},
                'process_noise_std': {'theta': 0.0, 'P': 100.0, 'lambda': 0.99},
                'lateral_slip_gain': {'theta': 0.0, 'P': 100.0, 'lambda': 0.99}
            }
        
        # Moving average parameters
        elif method == 'moving_average':
            self.ma_history = {
                'friction_coeff': deque(maxlen=window_size),
                'slippage_coeff': deque(maxlen=window_size),
                'process_noise_std': deque(maxlen=window_size),
                'lateral_slip_gain': deque(maxlen=window_size)
            }
    
    def add_observation(self, predicted_state, actual_state, control_input, state):
        """
        Add a new observation to the history
        """
        prediction_error = actual_state - predicted_state
        self.prediction_errors.append(prediction_error.copy())
        self.control_inputs.append(control_input.copy())
        self.states.append(state.copy())
        self.predicted_states.append(predicted_state.copy())
    
    def _estimate_friction_coeff(self):
        """
        Estimate friction coefficient from velocity prediction errors
        """
        if len(self.prediction_errors) < 5:
            return self.params['friction_coeff']
        
        errors = np.array(list(self.prediction_errors))
        inputs = np.array(list(self.control_inputs))
        
        # Friction affects velocity (state index 5) and acceleration (input index 0)
        # Error in velocity change should correlate with friction
        vel_errors = errors[:, 5]  # Velocity errors
        accel_inputs = inputs[:, 0]  # Acceleration commands
        
        if self.method == 'rls':
            # Use RLS to estimate friction from velocity errors vs acceleration
            if len(vel_errors) > 0 and np.any(np.abs(accel_inputs) > 0.01):
                # Simple linear model: vel_error = (1 - friction) * accel_effect
                # We estimate friction from how much velocity change we got vs expected
                valid_idx = np.abs(accel_inputs) > 0.01
                if np.any(valid_idx):
                    # Estimate: friction = 1 - (actual_vel_change / expected_vel_change)
                    # Simplified: use correlation between error and input
                    expected_change = accel_inputs[valid_idx] * 0.05  # dt approximation
                    actual_change = expected_change - vel_errors[valid_idx]
                    friction_est = np.clip(np.mean(actual_change / (expected_change + 1e-6)), 0.1, 1.0)
                    return 0.95 * self.params['friction_coeff'] + 0.05 * friction_est
            return self.params['friction_coeff']
        
        elif self.method == 'moving_average':
            # Estimate from recent observations
            if len(vel_errors) > 0 and np.any(np.abs(accel_inputs) > 0.01):
                valid_idx = np.abs(accel_inputs) > 0.01
                if np.any(valid_idx):
                    expected_change = accel_inputs[valid_idx] * 0.05
                    actual_change = expected_change - vel_errors[valid_idx]
                    friction_est = np.clip(np.mean(actual_change / (expected_change + 1e-6)), 0.1, 1.0)
                    self.ma_history['friction_coeff'].append(friction_est)
                    if len(self.ma_history['friction_coeff']) > 0:
                        return np.mean(self.ma_history['friction_coeff'])
            return self.params['friction_coeff']
        
        else:  # simple_estimation
            # Simple heuristic: if velocity errors are consistently negative when accelerating,
            # friction is lower than expected
            if len(vel_errors) > 10:
                recent_errors = vel_errors[-10:]
                recent_inputs = accel_inputs[-10:]
                accel_mask = recent_inputs > 0.1
                if np.any(accel_mask):
                    avg_error = np.mean(recent_errors[accel_mask])
                    if avg_error < -0.01:  # Less velocity than expected
                        return max(0.5, self.params['friction_coeff'] - 0.05)
                    elif avg_error > 0.01:  # More velocity than expected
                        return min(1.0, self.params['friction_coeff'] + 0.05)
            return self.params['friction_coeff']
    
    def _estimate_slippage_coeff(self):
        """
        Estimate slippage coefficient from heading/steering prediction errors
        """
        if len(self.prediction_errors) < 5:
            return self.params['slippage_coeff']
        
        errors = np.array(list(self.prediction_errors))
        inputs = np.array(list(self.control_inputs))
        states = np.array(list(self.states))
        
        # Slippage affects heading (state index 2) and steering rate (input index 1)
        heading_errors = errors[:, 2]  # Heading errors
        steering_inputs = inputs[:, 1]  # Steering rate commands
        velocities = states[:, 5]  # Velocities
        
        if self.method == 'rls':
            # Estimate slippage from heading change errors
            if len(heading_errors) > 0 and np.any(np.abs(steering_inputs) > 0.01):
                valid_idx = (np.abs(steering_inputs) > 0.01) & (np.abs(velocities) > 0.1)
                if np.any(valid_idx):
                    # Heading change should be proportional to steering and velocity
                    # Error indicates slippage
                    expected_change = steering_inputs[valid_idx] * 0.05  # dt approximation
                    actual_change = expected_change - heading_errors[valid_idx]
                    slippage_est = np.clip(np.mean(actual_change / (expected_change + 1e-6)), 0.1, 1.0)
                    return 0.95 * self.params['slippage_coeff'] + 0.05 * slippage_est
            return self.params['slippage_coeff']
        
        elif self.method == 'moving_average':
            if len(heading_errors) > 0 and np.any(np.abs(steering_inputs) > 0.01):
                valid_idx = (np.abs(steering_inputs) > 0.01) & (np.abs(velocities) > 0.1)
                if np.any(valid_idx):
                    expected_change = steering_inputs[valid_idx] * 0.05
                    actual_change = expected_change - heading_errors[valid_idx]
                    slippage_est = np.clip(np.mean(actual_change / (expected_change + 1e-6)), 0.1, 1.0)
                    self.ma_history['slippage_coeff'].append(slippage_est)
                    if len(self.ma_history['slippage_coeff']) > 0:
                        return np.mean(self.ma_history['slippage_coeff'])
            return self.params['slippage_coeff']
        
        else:  # simple_estimation
            if len(heading_errors) > 10:
                recent_errors = heading_errors[-10:]
                recent_inputs = steering_inputs[-10:]
                steering_mask = np.abs(recent_inputs) > 0.1
                if np.any(steering_mask):
                    avg_error = np.mean(np.abs(recent_errors[steering_mask]))
                    if avg_error > 0.05:  # Large heading errors indicate slippage
                        return max(0.5, self.params['slippage_coeff'] - 0.05)
                    elif avg_error < 0.01:  # Small errors, good traction
                        return min(1.0, self.params['slippage_coeff'] + 0.05)
            return self.params['slippage_coeff']
    
    def _estimate_process_noise_std(self):
        """
        Estimate process noise standard deviation from prediction error variance
        """
        if len(self.prediction_errors) < 5:
            return self.params['process_noise_std']
        
        errors = np.array(list(self.prediction_errors))
        
        # Process noise affects all states, estimate from error variance
        # Use recent errors to estimate noise level
        if len(errors) > 10:
            recent_errors = errors[-min(20, len(errors)):]
            # Estimate std from error magnitude (excluding systematic biases)
            error_std = np.std(recent_errors, axis=0)
            # Average across states, scale by dt
            noise_std = np.mean(error_std) / 0.05  # Divide by dt to get per-step noise
            
            if self.method == 'rls':
                return 0.95 * self.params['process_noise_std'] + 0.05 * noise_std
            elif self.method == 'moving_average':
                self.ma_history['process_noise_std'].append(noise_std)
                if len(self.ma_history['process_noise_std']) > 0:
                    return np.mean(self.ma_history['process_noise_std'])
            else:  # simple_estimation
                return np.clip(noise_std, 0.0, 0.1)
        
        return self.params['process_noise_std']
    
    def _estimate_lateral_slip_gain(self):
        """
        Estimate lateral slip gain from position errors perpendicular to heading
        """
        if len(self.prediction_errors) < 5:
            return self.params['lateral_slip_gain']
        
        errors = np.array(list(self.prediction_errors))
        states = np.array(list(self.states))
        
        # Lateral slip affects x, y positions (indices 0, 1)
        # Error perpendicular to heading indicates lateral drift
        pos_errors = errors[:, :2]  # x, y errors
        headings = states[:, 2]  # Headings
        
        if len(pos_errors) > 10:
            recent_pos_errors = pos_errors[-min(20, len(pos_errors)):]
            recent_headings = headings[-min(20, len(headings)):]
            
            # Compute lateral component of error (perpendicular to heading)
            lateral_errors = []
            for i in range(len(recent_pos_errors)):
                error_vec = recent_pos_errors[i]
                heading = recent_headings[i]
                # Rotate error to vehicle frame, lateral is y-component
                lateral = -error_vec[0] * np.sin(heading) + error_vec[1] * np.cos(heading)
                lateral_errors.append(abs(lateral))
            
            if len(lateral_errors) > 0:
                avg_lateral_error = np.mean(lateral_errors)
                # Estimate gain from lateral error magnitude
                # Scale by velocity and steering (if available)
                velocities = states[-len(lateral_errors):, 5]
                avg_vel = np.mean(np.abs(velocities))
                if avg_vel > 0.1:
                    slip_gain_est = avg_lateral_error / (avg_vel * 0.05 + 1e-6)  # Normalize by velocity*dt
                else:
                    slip_gain_est = avg_lateral_error / 0.05
                
                if self.method == 'rls':
                    return 0.95 * self.params['lateral_slip_gain'] + 0.05 * np.clip(slip_gain_est, 0.0, 0.1)
                elif self.method == 'moving_average':
                    self.ma_history['lateral_slip_gain'].append(np.clip(slip_gain_est, 0.0, 0.1))
                    if len(self.ma_history['lateral_slip_gain']) > 0:
                        return np.mean(self.ma_history['lateral_slip_gain'])
                else:  # simple_estimation
                    return np.clip(slip_gain_est, 0.0, 0.1)
        
        return self.params['lateral_slip_gain']
    
    def update(self):
        """
        Update all disturbance parameters based on current history
        """
        self.params['friction_coeff'] = self._estimate_friction_coeff()
        self.params['slippage_coeff'] = self._estimate_slippage_coeff()
        self.params['process_noise_std'] = self._estimate_process_noise_std()
        self.params['lateral_slip_gain'] = self._estimate_lateral_slip_gain()
        
        # Clip parameters to reasonable ranges
        self.params['friction_coeff'] = np.clip(self.params['friction_coeff'], 0.1, 1.0)
        self.params['slippage_coeff'] = np.clip(self.params['slippage_coeff'], 0.1, 1.0)
        self.params['process_noise_std'] = np.clip(self.params['process_noise_std'], 0.0, 0.1)
        self.params['lateral_slip_gain'] = np.clip(self.params['lateral_slip_gain'], 0.0, 0.1)
        
        return self.params.copy()
    
    def get_params(self):
        """Get current estimated parameters"""
        return self.params.copy()


class MPCTrackingControlOnline(TrajectoryPlanning):
    """
    Online adaptive MPC with disturbance learning and adaptive constraints
    """
    def __init__(self, 
                 dynamics, params, 
                 Q, R, 
                 state_bound, input_bound,
                 learning_method='rls',
                 window_size=50,
                 initial_disturbance_params=None,
                 adaptive_constraint_tightening=True):
        super().__init__(dynamics, params, Q, R, state_bound, input_bound)
        
        self._ref_states_sm = ca.SX.sym("reference_states", self._num_state, self._horizon+1)
        self._ref_inputs_sm = ca.SX.sym("reference_inputs", self._num_input, self._horizon)
        
        # Initialize disturbance estimator
        self.estimator = OnlineDisturbanceEstimator(
            method=learning_method,
            window_size=window_size,
            initial_params=initial_disturbance_params
        )
        
        self.adaptive_constraint_tightening = adaptive_constraint_tightening
        
        # Store last predicted state for error computation
        self.last_predicted_state = None
        self.last_control_input = None
        
        self._build_solver()
    
    def _cost_function(self):
        cost = 0.
        for k in range(self._horizon):
            cost += (self._inputs_sm[:,k] - self._ref_inputs_sm[:,k]).T @ self._R @ (self._inputs_sm[:,k] - self._ref_inputs_sm[:,k]) + \
                (self._states_sm[:,k] - self._ref_states_sm[:,k]).T  @ self._Q @ (self._states_sm[:,k] - self._ref_states_sm[:,k])
        Q_f = self._Q
        cost += (self._states_sm[:,-1] - self._ref_states_sm[:,-1]).T  @ Q_f @ (self._states_sm[:,-1] - self._ref_states_sm[:,-1])
        
        return cost
    
    def _compute_adaptive_constraints(self):
        """
        Compute adaptive constraint bounds based on learned disturbance parameters
        """
        params = self.estimator.get_params()
        
        # Base constraints
        lb_vars_base = []
        ub_vars_base = []
        
        # Adaptive tightening based on uncertainty
        # Higher uncertainty -> tighter constraints (more conservative)
        uncertainty_factor = 1.0 + params['process_noise_std'] * 10.0
        
        # Friction affects velocity bounds (lower friction -> more conservative)
        friction_factor = 1.0 / (params['friction_coeff'] + 0.1)  # Inverse relationship
        
        # Slippage affects steering bounds (more slippage -> more conservative)
        slippage_factor = 1.0 / (params['slippage_coeff'] + 0.1)
        
        for k in range(self._horizon):
            # State constraints with adaptive tightening
            state_lb = self._state_bound['lb']
            state_ub = self._state_bound['ub']
            
            # Tighten velocity bounds based on friction
            if self.adaptive_constraint_tightening:
                # Reduce max velocity when friction is low
                v_max_tightened = float(state_ub[5]) * params['friction_coeff']
                v_min_tightened = float(state_lb[5]) * params['friction_coeff']
                state_ub_adapted = ca.DM(state_ub)
                state_lb_adapted = ca.DM(state_lb)
                state_ub_adapted[5] = min(float(state_ub[5]), v_max_tightened)
                state_lb_adapted[5] = max(float(state_lb[5]), v_min_tightened)
                
                # Tighten steering angle bounds based on slippage
                phi_max_tightened = float(state_ub[4]) * params['slippage_coeff']
                phi_min_tightened = float(state_lb[4]) * params['slippage_coeff']
                state_ub_adapted[4] = min(float(state_ub[4]), phi_max_tightened)
                state_lb_adapted[4] = max(float(state_lb[4]), phi_min_tightened)
            else:
                state_ub_adapted = state_ub
                state_lb_adapted = state_lb
            
            # Input constraints with adaptive tightening
            input_lb = self._input_bound['lb']
            input_ub = self._input_bound['ub']
            
            if self.adaptive_constraint_tightening:
                # Reduce acceleration bounds when friction is low
                a_max_tightened = float(input_ub[0]) * params['friction_coeff']
                a_min_tightened = float(input_lb[0]) * params['friction_coeff']
                input_ub_adapted = ca.DM(input_ub)
                input_lb_adapted = ca.DM(input_lb)
                input_ub_adapted[0] = min(float(input_ub[0]), a_max_tightened)
                input_lb_adapted[0] = max(float(input_lb[0]), a_min_tightened)
                
                # Reduce steering rate bounds when slippage is high
                omega_max_tightened = float(input_ub[1]) * params['slippage_coeff']
                omega_min_tightened = float(input_lb[1]) * params['slippage_coeff']
                input_ub_adapted[1] = min(float(input_ub[1]), omega_max_tightened)
                input_lb_adapted[1] = max(float(input_lb[1]), omega_min_tightened)
            else:
                input_ub_adapted = input_ub
                input_lb_adapted = input_lb
            
            lb_vars_base.append(state_lb_adapted)
            lb_vars_base.append(input_lb_adapted)
            ub_vars_base.append(state_ub_adapted)
            ub_vars_base.append(input_ub_adapted)
        
        # Final state
        state_lb = self._state_bound['lb']
        state_ub = self._state_bound['ub']
        if self.adaptive_constraint_tightening:
            state_ub_adapted = ca.DM(state_ub)
            state_lb_adapted = ca.DM(state_lb)
            v_max_tightened = float(state_ub[5]) * params['friction_coeff']
            v_min_tightened = float(state_lb[5]) * params['friction_coeff']
            state_ub_adapted[5] = min(float(state_ub[5]), v_max_tightened)
            state_lb_adapted[5] = max(float(state_lb[5]), v_min_tightened)
            phi_max_tightened = float(state_ub[4]) * params['slippage_coeff']
            phi_min_tightened = float(state_lb[4]) * params['slippage_coeff']
            state_ub_adapted[4] = min(float(state_ub[4]), phi_max_tightened)
            state_lb_adapted[4] = max(float(state_lb[4]), phi_min_tightened)
        else:
            state_ub_adapted = state_ub
            state_lb_adapted = state_lb
        
        lb_vars_base.append(state_lb_adapted)
        ub_vars_base.append(state_ub_adapted)
        
        lb_vars = ca.vertcat(*lb_vars_base)
        ub_vars = ca.vertcat(*ub_vars_base)
        
        return lb_vars, ub_vars
        
    def _state_input_constraints(self):
        # This will be called during build, but we'll recompute adaptively in solve
        self._vars = []
        lb_vars = []
        ub_vars = []
        for k in range(self._horizon):
            self._vars = ca.vertcat(self._vars,
                                    self._states_sm[:,k],
                                    self._inputs_sm[:,k])
            lb_vars = ca.vertcat(lb_vars,
                                 self._state_bound['lb'],
                                 self._input_bound['lb'])
            ub_vars = ca.vertcat(ub_vars, 
                                 self._state_bound['ub'], 
                                 self._input_bound['ub'])
        
        self._vars = ca.vertcat(self._vars,
                                self._states_sm[:,-1])
        lb_vars = ca.vertcat(lb_vars,
                             self._state_bound['lb'])
        ub_vars = ca.vertcat(ub_vars,
                             self._state_bound['ub'])
        
        return lb_vars, ub_vars
    
    def _build_solver(self):
        g_dyn, lb_dyn, ub_dyn = self._dynamics_constraints()
        g = g_dyn
        self._lb_g = lb_dyn
        self._ub_g = ub_dyn
                        
        self._lb_vars_base, self._ub_vars_base = self._state_input_constraints()        
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
              reference_inputs,
              actual_state=None,
              update_estimator=True):
        """
        Solve MPC with online adaptation
        
        Args:
            initial_state: Current state
            reference_states: Reference trajectory
            reference_inputs: Reference inputs
            actual_state: Previous actual state (for learning, if available)
            update_estimator: Whether to update disturbance parameters
        """
        # Update disturbance estimator if we have previous prediction
        if update_estimator and actual_state is not None and self.last_predicted_state is not None and self.last_control_input is not None:
            self.estimator.add_observation(
                self.last_predicted_state,
                actual_state,
                self.last_control_input,
                initial_state  # Use current state as context
            )
            # Update parameters every step
            self.estimator.update()
        
        # Compute adaptive constraints based on learned parameters
        if self.adaptive_constraint_tightening:
            lb_vars, ub_vars = self._compute_adaptive_constraints()
        else:
            lb_vars, ub_vars = self._lb_vars_base, self._ub_vars_base
        
        reference_states_flat = reference_states.T.reshape((-1, 1))
        reference_inputs_flat = reference_inputs.T.reshape((-1, 1))
        
        vars_guess = self._get_initial_guess(reference_states, reference_inputs)
        start = time.time()
        sol = self._solver(
                x0  = vars_guess, 
                lbx = lb_vars,
                ubx = ub_vars,
                lbg = self._lb_g,
                ubg = self._ub_g,
                p = ca.vertcat(reference_states_flat, 
                               reference_inputs_flat,
                               initial_state)
            )
        end = time.time()
        
        vars_opt = sol['x']
        
        if self._solver.stats()['success'] == False:
            print("Cannot find a solution!") 
        
        states, inputs = self._split_decision_variables(vars_opt)
        
        # Store prediction for next step's learning
        self.last_predicted_state = states[:, 1].copy()  # Next predicted state
        self.last_control_input = inputs[:, 0].copy()    # First control input
        
        return states, inputs
    
    def get_disturbance_params(self):
        """Get current learned disturbance parameters"""
        return self.estimator.get_params()
    
    def reset_estimator(self, initial_params=None):
        """Reset the disturbance estimator"""
        self.estimator = OnlineDisturbanceEstimator(
            method=self.estimator.method,
            window_size=self.estimator.window_size,
            initial_params=initial_params
        )

