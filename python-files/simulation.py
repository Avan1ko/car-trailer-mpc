import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Rectangle
from math import cos, sin, tan, pi
from draw import draw_truck_trailer
import math
from truck_trailer_model import TruckTrailerModel
from mpc_control import MPCTrackingControl
from mpc_control_obs import MPCTrackingControlObs
from get_obstacles import get_obstacles
from get_initial_goal_states import get_initial_goal_states
import casadi as ca
import copy
import os
from LQR_cost import lqr_distance

# ============================================================================
# DISTURBANCE CONFIGURATION
# ============================================================================
ENABLE_DISTURBANCES = True  # Set to False to disable all disturbances
USE_OBS_MPC = False  # Set to True to use MPC with obstacle constraints

DISTURBANCE_PARAMS = {
    'friction_coeff': .9,        # .7 0.0-1.0, 1.0 = perfect friction, 0.7 = 70% friction (low friction)
    'slippage_coeff': .9,        # .8 0.0-1.0, 1.0 = no slippage, 0.8 = 80% steering effectiveness (20% slippage)
    'process_noise_std': 0.02,     # .02 Standard deviation for process noise (additive to states)
    'lateral_slip_gain': 0.00,     # Lateral drift coefficient (sideways movement)
    'slip_angle_max': 0.0,         # Maximum slip angle in radians for tire slippage model
}

def f_dyn(q, u, params):
    L1 = params['L1']
    L2 = params['L2']
    M  = params['M']
    x, y, theta, psi, phi, v = q
    a, omega = u
    q_dot = np.zeros(q.shape)
    q_dot[0] = v * cos(theta)
    q_dot[1] = v * sin(theta)
    q_dot[2] = v * tan(phi) / L1
    q_dot[3] = - v * tan(phi) / L1 * (1 + M/L2 * cos(psi)) - v * sin(psi) / L2
    q_dot[4] = omega
    q_dot[5] = a
    
    return q_dot

def apply_disturbances(q, u, params, disturbance_params=None):
    """
    Apply various disturbances to the system inputs and generate process noise
    
    Args:
        q: current state [x, y, theta, psi, phi, v]
        u: control input [acceleration, steering_rate]
        params: system parameters
        disturbance_params: dict with disturbance parameters
        
    Returns:
        u_disturbed: modified control input with disturbances applied
        state_noise: additive process noise vector
    """
    if disturbance_params is None:
        return u.copy(), np.zeros(6)
    
    u_disturbed = u.copy()
    state_noise = np.zeros(6)
    
    # Friction affects acceleration effectiveness
    if 'friction_coeff' in disturbance_params:
        mu = disturbance_params['friction_coeff']
        # Reduce acceleration effectiveness due to low friction
        u_disturbed[0] *= mu  # Acceleration command reduced
    
    # Slippage affects steering effectiveness
    if 'slippage_coeff' in disturbance_params:
        eta = disturbance_params['slippage_coeff']
        # Reduce steering rate effectiveness due to tire slippage
        u_disturbed[1] *= eta  # Steering rate command reduced
    
    # Process noise (modeling uncertainty, sensor noise, etc.)
    if 'process_noise_std' in disturbance_params:
        std = disturbance_params['process_noise_std']
        state_noise = np.random.normal(0, std, 6)
    
    return u_disturbed, state_noise

def apply_slippage_to_dynamics(q_dot, q, params, disturbance_params=None):
    """
    Apply tire slippage effects to the dynamics (affects how vehicle turns)
    
    Args:
        q_dot: nominal state derivative
        q: current state
        params: system parameters
        disturbance_params: dict with disturbance parameters
        
    Returns:
        q_dot_modified: dynamics with slippage applied
    """
    if disturbance_params is None or 'slip_angle_max' not in disturbance_params:
        return q_dot
    
    phi = q[4]  # Steering angle
    v = q[5]    # Velocity
    
    # Slip increases with speed and steering angle magnitude
    # Higher speed and sharper turns = more slippage
    slip_factor = 1.0 - min(abs(phi) * abs(v) * disturbance_params['slip_angle_max'], 0.3)
    
    # Apply slippage to heading and hitch angle changes (turning dynamics)
    q_dot[2] *= slip_factor  # Heading change reduced
    q_dot[3] *= slip_factor  # Hitch angle change reduced
    
    return q_dot

def apply_lateral_slip(q, q_dot, params, disturbance_params=None):
    """
    Apply lateral slip (sideways drift) to the position update
    
    Args:
        q: current state
        q_dot: state derivative
        params: system parameters
        disturbance_params: dict with disturbance parameters
        
    Returns:
        lateral_drift: [dx, dy] lateral drift components
    """
    if disturbance_params is None or 'lateral_slip_gain' not in disturbance_params:
        return np.array([0.0, 0.0])
    
    slip_gain = disturbance_params['lateral_slip_gain']
    v = q[5]      # Velocity
    phi = q[4]    # Steering angle
    theta = q[2]  # Heading
    
    # Lateral drift proportional to speed and steering angle
    # More steering and higher speed = more lateral drift
    lateral_drift_magnitude = slip_gain * abs(v) * abs(phi)
    
    # Drift direction is perpendicular to heading (90 degrees)
    lateral_drift = np.array([
        lateral_drift_magnitude * cos(theta + pi/2),
        lateral_drift_magnitude * sin(theta + pi/2)
    ])
    
    return lateral_drift

def update(q, u, params, disturbance_params=None):
    """
    Update state with optional disturbances
    
    Args:
        q: current state [x, y, theta, psi, phi, v]
        u: control input [acceleration, steering_rate]
        params: system parameters
        disturbance_params: dict with disturbance parameters (None to disable)
        
    Returns:
        q_: updated state
    """
    # Apply disturbances to control inputs and generate process noise
    u_disturbed, state_noise = apply_disturbances(q, u, params, disturbance_params)
    
    # Compute nominal dynamics
    q_dot = f_dyn(q, u_disturbed, params)
    
    # Apply slippage to dynamics (affects turning)
    if disturbance_params is not None:
        q_dot = apply_slippage_to_dynamics(q_dot, q, params, disturbance_params)
    
    # Update state with dynamics
    q_ = q + q_dot * params["dt"]
    
    # Add process noise
    q_ += state_noise * params["dt"]
    
    # Apply lateral slip (sideways drift)
    if disturbance_params is not None:
        lateral_drift = apply_lateral_slip(q, q_dot, params, disturbance_params)
        q_[0] += lateral_drift[0] * params["dt"]  # x position
        q_[1] += lateral_drift[1] * params["dt"]  # y position
    
    return q_

def do_interpolation(state_traj, input_traj, dt_1, dt_2):
    # from dt_1 to dt_2
    # dt_1 > dt_2
    N = input_traj.shape[1]
    num_state = state_traj.shape[0]
    num_input = input_traj.shape[0]
    n = math.floor(dt_1/dt_2)
    N_new = n*N
    state_traj_new = np.zeros((num_state, N_new+1))
    input_traj_new = np.zeros((num_input, N_new))
    for k in range(N):
        for m in range(n):
            t = m/n
            state_traj_new[:,k*n+m] = (1-t)*state_traj[:,k] + t*state_traj[:,k+1]
            input_traj_new[:,k*n+m] = input_traj[:,k] 
    state_traj_new[:,-1] = state_traj[:,-1]
    
    return state_traj_new, input_traj_new 

if __name__ == "__main__":
    dt_to = 0.1
    dt = 0.05
    horizon = 60
    params = {"M": 0.15, 
              "L1": 7.05, "L2": 12.45, 
              "W1": 3.05, "W2": 2.95,
              "dt": dt,
              "horizon": horizon}
            
    model = TruckTrailerModel(params)
    #obstacle_list = [{"center": (5, -5), "width": 10, "height": 10}]
    obstacle_list = get_obstacles()
    Q = np.eye(6)
    Q[0, 0] = 1.  # x position 
    Q[1, 1] = 1.  # y position 
    Q[2, 2] = 1.  # truck heading 
    Q[3, 3] = 1.  # hitch angle 
    Q[4, 4] = 1.  # steering angle 
    Q[5, 5] = 1.  # speed
    R = np.eye(2)
    R[0, 0] = 10.  # acceleration 
    R[1, 1] = 10.  # steering speed 
    # (x, y, theta, psi, phi, v)
    state_bound = {"lb": ca.DM([-ca.inf, -ca.inf, -ca.pi, -ca.pi/3., -ca.pi/4., -10.]),
                   "ub": ca.DM([ ca.inf,  ca.inf,  ca.pi,  ca.pi/3.,  ca.pi/4.,  10.])}      
    input_bound = {"lb": ca.DM([-5, -ca.pi/2]),
                   "ub": ca.DM([ 5,  ca.pi/2])}
    if USE_OBS_MPC:
        controller = MPCTrackingControlObs(model, params,
                                    Q, R, 
                                    state_bound, input_bound, obstacle_list=obstacle_list)
    else:
        controller = MPCTrackingControl(model, params,
                                    Q, R, 
                                    state_bound, input_bound)
    initial_state, goal_state = get_initial_goal_states()
    initial_state.append(0)
    initial_state.append(0)
    initial_state = np.array(initial_state)
    # initial_state = np.array([0., 3., 0., 0., 0., 0.])
   

    dir_path = os.path.dirname(os.path.realpath(__file__))
    ref_state_traj = np.loadtxt(dir_path+'/data/state_traj.txt')
    ref_input_traj = np.loadtxt(dir_path+'/data/input_traj.txt')
    
    ref_state_traj, ref_input_traj = do_interpolation(ref_state_traj, ref_input_traj, dt_to, dt)
    
    T_sim = 40.
    # T_sim = 10. / 30
    t = 0.
    
    state = copy.deepcopy(initial_state)
    x =  [state[0]]
    y = [state[1]]
    theta = [state[2]]
    psi = [state[3]]
    phi = [state[4]]
    v = [state[5]]

    ref_state_traj_ = np.zeros((model.num_state, horizon+1))
    ref_input_traj_ = np.zeros((model.num_input, horizon))
    N = ref_input_traj.shape[1]
    
    # Print disturbance status
    if ENABLE_DISTURBANCES:
        print("=" * 60)
        print("DISTURBANCES ENABLED")
        print(f"  Friction coefficient: {DISTURBANCE_PARAMS['friction_coeff']}")
        print(f"  Slippage coefficient: {DISTURBANCE_PARAMS['slippage_coeff']}")
        print(f"  Process noise std: {DISTURBANCE_PARAMS['process_noise_std']}")
        print(f"  Lateral slip gain: {DISTURBANCE_PARAMS['lateral_slip_gain']}")
        print("=" * 60)
    else:
        print("Disturbances DISABLED - nominal simulation")
    
    solve_times = []
    
    while t <= T_sim:
        k = math.floor(t/dt)
        if k + horizon <= N:
            ref_state_traj_[:,:] = ref_state_traj[:,k:k+horizon+1]
            ref_input_traj_[:,:] = ref_input_traj[:,k:k+horizon]
        elif k < N:
            print(f"horizon: {horizon}")
            print(f"k: {k}")
            print(f"N: {N}")
            ref_state_traj_[:,:N+1-k] = ref_state_traj[:,k:]
            ref_state_traj_[:,N+1-k:] = ca.repmat(ref_state_traj[:,-1], 1, horizon-N+k)
            ref_input_traj_[:,:N-k] = ref_input_traj[:,k:]
            ref_input_traj_[:,N-k:] = ca.repmat(ref_input_traj[:,-1], 1, horizon-N+k)
        else:
            ref_state_traj_[:,:] = ca.repmat(ref_state_traj[:,-1], 1, horizon+1)
            ref_input_traj_[:,:] = ca.repmat(np.zeros((model.num_input,1)), 1, horizon)
        
        t_start = time.perf_counter()
        states, inputs = controller.solve(state, ref_state_traj_, ref_input_traj_)
        t_end = time.perf_counter()
        solve_times.append(t_end - t_start)
        u_con = inputs[:,0]
        
        # Apply disturbances if enabled
        if ENABLE_DISTURBANCES:
            state = update(state, u_con, params, disturbance_params=DISTURBANCE_PARAMS)
        else:
            state = update(state, u_con, params)
        
        x.append(state[0])
        y.append(state[1])
        theta.append(state[2])
        psi.append(state[3])
        phi.append(state[4])
        v.append(state[5])

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(ref_state_traj[0,:], ref_state_traj[1,:], "-r", label="course")
        plt.plot(x, y, "ob", label="trajectory")
        for obstacle in obstacle_list:
            w = obstacle["width"]
            h = obstacle["height"]
            c = obstacle["center"]
            xy = (c[0]-w/2, c[1]-h/2)
            plt.gca().add_patch(Rectangle(xy, w, h))
        plt.plot(states[0,:], states[1,:], '-o', color='darkorange')
        # plt.plot(ref_state_traj[0,k:k+horizon+1], ref_state_traj[1,k:k+horizon+1], '-o')
        plt.plot(ref_state_traj_[0,:], ref_state_traj_[1,:], '-o')
        draw_truck_trailer(pose=state[:4], params=params)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)
    
        t += dt
    
    #calculate the "distance" of the final state to the desired goal state using LQR
    score = lqr_distance(state[-6:], ref_state_traj_[:, -1], params, model, Q, R, ref_input_traj_[:, -1])
    print("LQR distance score: ")
    print(score)
    
    # Print solve time statistics
    print("SOLVE TIME STATISTICS")
    print(f"  Min solve time:  {min(solve_times)*1000:.3f} ms")
    print(f"  Max solve time:  {max(solve_times)*1000:.3f} ms")
    print(f"  Avg solve time:  {np.mean(solve_times)*1000:.3f} ms")
    
    plt.show()
    