import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import cos, sin, tan, pi
from draw import draw_truck_trailer
import math
from truck_trailer_model import TruckTrailerModel
from mpc_control_nmpc import TruckTrailerNMPC
from get_obstacles import get_obstacles
from get_initial_goal_states import get_initial_goal_states
import casadi as ca
import copy
import os

# ============================================================================
# DISTURBANCE CONFIGURATION
# ============================================================================
ENABLE_DISTURBANCES = False

DISTURBANCE_PARAMS = {
    'friction_coeff': 1,
    'slippage_coeff': 1,
    'process_noise_std': 0.02,
    'lateral_slip_gain': 0.00,
    'slip_angle_max': 0.0,
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
    if disturbance_params is None:
        return u.copy(), np.zeros(6)
    
    u_disturbed = u.copy()
    state_noise = np.zeros(6)
    
    if 'friction_coeff' in disturbance_params:
        mu = disturbance_params['friction_coeff']
        u_disturbed[0] *= mu
    
    if 'slippage_coeff' in disturbance_params:
        eta = disturbance_params['slippage_coeff']
        u_disturbed[1] *= eta
    
    if 'process_noise_std' in disturbance_params:
        std = disturbance_params['process_noise_std']
        state_noise = np.random.normal(0, std, 6)
    
    return u_disturbed, state_noise

def apply_slippage_to_dynamics(q_dot, q, params, disturbance_params=None):
    if disturbance_params is None or 'slip_angle_max' not in disturbance_params:
        return q_dot
    
    phi = q[4]
    v = q[5]
    slip_factor = 1.0 - min(abs(phi) * abs(v) * disturbance_params['slip_angle_max'], 0.3)
    q_dot[2] *= slip_factor
    q_dot[3] *= slip_factor
    
    return q_dot

def apply_lateral_slip(q, q_dot, params, disturbance_params=None):
    if disturbance_params is None or 'lateral_slip_gain' not in disturbance_params:
        return np.array([0.0, 0.0])
    
    slip_gain = disturbance_params['lateral_slip_gain']
    v = q[5]
    phi = q[4]
    theta = q[2]
    lateral_drift_magnitude = slip_gain * abs(v) * abs(phi)
    lateral_drift = np.array([
        lateral_drift_magnitude * cos(theta + pi/2),
        lateral_drift_magnitude * sin(theta + pi/2)
    ])
    
    return lateral_drift

def update(q, u, params, disturbance_params=None):
    u_disturbed, state_noise = apply_disturbances(q, u, params, disturbance_params)
    q_dot = f_dyn(q, u_disturbed, params)
    if disturbance_params is not None:
        q_dot = apply_slippage_to_dynamics(q_dot, q, params, disturbance_params)
    q_ = q + q_dot * params["dt"]
    q_ += state_noise * params["dt"]
    if disturbance_params is not None:
        lateral_drift = apply_lateral_slip(q, q_dot, params, disturbance_params)
        q_[0] += lateral_drift[0] * params["dt"]
        q_[1] += lateral_drift[1] * params["dt"]
    return q_

def do_interpolation(state_traj, input_traj, dt_1, dt_2):
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
    horizon = 30  # shorter for NMPC stability/runtime
    params = {"M": 0.15, 
              "L1": 7.05, "L2": 12.45, 
              "W1": 3.05, "W2": 2.95,
              "dt": dt,
              "horizon": horizon}
            
    model = TruckTrailerModel(params)
    obstacle_list = get_obstacles()
    Q = np.eye(6)
    Q[0, 0] = 1.
    Q[1, 1] = 1.
    Q[2, 2] = 2.  # heading tighter
    Q[3, 3] = 3.  # hitch tighter
    Q[4, 4] = 1.
    Q[5, 5] = 1.
    R = np.eye(2)
    R[0, 0] = 5.
    R[1, 1] = 8.
    state_bound = {"lb": ca.DM([-ca.inf, -ca.inf, -ca.pi, -ca.pi/3., -ca.pi/4., -8.]),
                   "ub": ca.DM([ ca.inf,  ca.inf,  ca.pi,  ca.pi/3.,  ca.pi/4.,  8.])}
    input_bound = {"lb": ca.DM([-4, -ca.pi/2]),
                   "ub": ca.DM([ 4,  ca.pi/2])}
    controller = TruckTrailerNMPC(model, params,
                                  Q, R, 
                                  state_bound, input_bound)
    initial_state, goal_state = get_initial_goal_states()
    initial_state.append(0)
    initial_state.append(0)
    initial_state = np.array(initial_state)
   
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ref_state_traj = np.loadtxt(dir_path+'/data/state_traj.txt')
    ref_input_traj = np.loadtxt(dir_path+'/data/input_traj.txt')
    
    ref_state_traj, ref_input_traj = do_interpolation(ref_state_traj, ref_input_traj, dt_to, dt)
    
    T_sim = 25.0  # shorter to keep NMPC runtime reasonable
    time = 0.
    
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
    last_control = np.zeros(model.num_input)
    failure_count = 0
    consecutive_failures = 0
    
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
    
    while time <= T_sim:
        k = math.floor(time/dt)
        if k + horizon <= N:
            ref_state_traj_[:,:] = ref_state_traj[:,k:k+horizon+1]
            ref_input_traj_[:,:] = ref_input_traj[:,k:k+horizon]
        elif k < N:
            ref_state_traj_[:,:N+1-k] = ref_state_traj[:,k:]
            ref_state_traj_[:,N+1-k:] = ca.repmat(ref_state_traj[:,-1], 1, horizon-N+k)
            ref_input_traj_[:,:N-k] = ref_input_traj[:,k:]
            ref_input_traj_[:,N-k:] = ca.repmat(ref_input_traj[:,-1], 1, horizon-N+k)
        else:
            ref_state_traj_[:,:] = ca.repmat(ref_state_traj[:,-1], 1, horizon+1)
            ref_input_traj_[:,:] = ca.repmat(np.zeros((model.num_input,1)), 1, horizon)
        
        states, inputs = controller.solve(state, ref_state_traj_, ref_input_traj_)
        if states is None or inputs is None:
            failure_count += 1
            consecutive_failures += 1
            if failure_count <= 5 or failure_count % 20 == 0:
                print("NMPC failed, zeroing control.")
            u_con = np.zeros(model.num_input)
            last_control = u_con
            if consecutive_failures > 20:
                print("Too many consecutive NMPC failures, stopping simulation.")
                break
        else:
            u_con = inputs[:,0]
            last_control = u_con
            consecutive_failures = 0
        
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
        if states is not None:
            plt.plot(states[0,:], states[1,:], '-o', color='darkorange')
        else:
            plt.plot(ref_state_traj_[0,:], ref_state_traj_[1,:], '--', color='darkorange')
        plt.plot(ref_state_traj_[0,:], ref_state_traj_[1,:], '-o')
        draw_truck_trailer(pose=state[:4], params=params)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)
    
        time += dt
    
    plt.show()
