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
# Note: Controller switches dynamically based on obstacle collision detection in MPC horizon
USE_OBS_MPC = False  # Set to True to use MPC with obstacle constraints
USE_SWITCH_MPC = False  # Set to True to use switch MPC

DISTURBANCE_PARAMS = {
    'friction_coeff': .9,        # .7 0.0-1.0, 1.0 = perfect friction, 0.7 = 70% friction (low friction)
    'slippage_coeff': .9,        # .8 0.0-1.0, 1.0 = no slippage, 0.8 = 80% steering effectiveness (20% slippage)
    'process_noise_std': 0.02,     # .02 Standard deviation for process noise (additive to states)
    'lateral_slip_gain': 0.01,     # Lateral drift coefficient (sideways movement)
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

def generate_measurement_noise(disturbance_params):
    """
    Generate measurement noise for the state passed to MPC.
    This simulates sensor uncertainty where the controller doesn't know the true state.
    
    Args:
        disturbance_params: dict with disturbance parameters
        
    Returns:
        measurement_noise: additive noise vector for state measurement
    """
    if disturbance_params is None or 'process_noise_std' not in disturbance_params:
        return np.zeros(6)
    std = disturbance_params['process_noise_std']
    return np.random.normal(0, std, 6)

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
    # Apply disturbances to control inputs (friction, slippage on actuators)
    u_disturbed, _ = apply_disturbances(q, u, params, disturbance_params)
    
    # Compute nominal dynamics
    q_dot = f_dyn(q, u_disturbed, params)
    
    # Apply slippage to dynamics (affects turning)
    if disturbance_params is not None:
        q_dot = apply_slippage_to_dynamics(q_dot, q, params, disturbance_params)
    
    # Update state with dynamics (no process noise - it's applied as measurement noise to MPC)
    q_ = q + q_dot * params["dt"]
    
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

# ============================================================================
# COLLISION DETECTION HELPERS
# ============================================================================

def get_rect_corners(center_x, center_y, half_length, half_width, angle):
    """
    Get the 4 corners of an oriented rectangle.
    
    Args:
        center_x, center_y: Center position of rectangle
        half_length, half_width: Half dimensions (length along heading, width perpendicular)
        angle: Rotation angle in radians
        
    Returns:
        corners: numpy array of shape (4, 2) with corner coordinates
    """
    c, s = np.cos(angle), np.sin(angle)
    # Local corners (centered at origin)
    local_corners = np.array([
        [ half_length,  half_width],
        [ half_length, -half_width],
        [-half_length, -half_width],
        [-half_length,  half_width]
    ])
    # Rotation matrix
    R = np.array([[c, -s], [s, c]])
    # Rotate and translate
    corners = (R @ local_corners.T).T + np.array([center_x, center_y])
    return corners

def project_polygon(corners, axis):
    """Project polygon corners onto an axis and return min/max."""
    projections = corners @ axis
    return projections.min(), projections.max()

def check_obb_aabb_collision(obb_corners, aabb_center, aabb_half_w, aabb_half_h):
    """
    Check collision between an oriented bounding box (OBB) and an axis-aligned 
    bounding box (AABB) using the Separating Axis Theorem.
    
    Args:
        obb_corners: numpy array (4, 2) of OBB corner coordinates
        aabb_center: tuple (cx, cy) of AABB center
        aabb_half_w, aabb_half_h: Half width and height of AABB
        
    Returns:
        True if collision detected, False otherwise
    """
    # AABB corners
    cx, cy = aabb_center
    aabb_corners = np.array([
        [cx + aabb_half_w, cy + aabb_half_h],
        [cx + aabb_half_w, cy - aabb_half_h],
        [cx - aabb_half_w, cy - aabb_half_h],
        [cx - aabb_half_w, cy + aabb_half_h]
    ])
    
    # Get axes to test (normals of edges)
    # For AABB: always (1,0) and (0,1)
    # For OBB: perpendicular to first two edges
    axes = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]
    
    # OBB edge normals
    edge1 = obb_corners[1] - obb_corners[0]
    edge2 = obb_corners[3] - obb_corners[0]
    # Normalize and get perpendiculars
    if np.linalg.norm(edge1) > 1e-9:
        axes.append(np.array([-edge1[1], edge1[0]]) / np.linalg.norm(edge1))
    if np.linalg.norm(edge2) > 1e-9:
        axes.append(np.array([-edge2[1], edge2[0]]) / np.linalg.norm(edge2))
    
    # Test all axes
    for axis in axes:
        obb_min, obb_max = project_polygon(obb_corners, axis)
        aabb_min, aabb_max = project_polygon(aabb_corners, axis)
        
        # Check for gap (separating axis found)
        if obb_max < aabb_min or aabb_max < obb_min:
            return False
    
    # No separating axis found - collision!
    return True

def get_vehicle_center_np(x_rear, y_rear, heading, params):
    """Compute vehicle center position (numpy version)."""
    xv = x_rear + np.cos(heading) * params['L1'] / 2
    yv = y_rear + np.sin(heading) * params['L1'] / 2
    return xv, yv

def get_trailer_center_np(x_rear, y_rear, heading, psi, params):
    """Compute trailer center position (numpy version)."""
    x_hitch = x_rear - np.cos(heading) * params['M']
    y_hitch = y_rear - np.sin(heading) * params['M']
    xt = x_hitch - np.cos(heading + psi) * params['L2'] / 2
    yt = y_hitch - np.sin(heading + psi) * params['L2'] / 2
    return xt, yt

def check_state_collision(state, params, obstacle_list):
    """
    Check if a single state (vehicle + trailer) collides with any obstacle.
    
    Args:
        state: State vector [x, y, theta, psi, phi, v] or at least [x, y, theta, psi]
        params: Vehicle parameters dict
        obstacle_list: List of obstacles with center, width, height
        
    Returns:
        True if collision detected
    """
    x_rear, y_rear, heading, psi = state[0], state[1], state[2], state[3]
    
    # Get vehicle geometry
    veh_cx, veh_cy = get_vehicle_center_np(x_rear, y_rear, heading, params)
    veh_half_l = params['L1'] / 2
    veh_half_w = params['W1'] / 2
    veh_corners = get_rect_corners(veh_cx, veh_cy, veh_half_l, veh_half_w, heading)
    
    # Get trailer geometry
    trl_cx, trl_cy = get_trailer_center_np(x_rear, y_rear, heading, psi, params)
    trl_half_l = params['L2'] / 2
    trl_half_w = params['W2'] / 2
    trailer_heading = heading + psi
    trl_corners = get_rect_corners(trl_cx, trl_cy, trl_half_l, trl_half_w, trailer_heading)
    
    # Check against each obstacle
    for obs in obstacle_list:
        obs_center = obs['center']
        obs_half_w = obs['width'] / 2
        obs_half_h = obs['height'] / 2
        
        # Check vehicle collision
        if check_obb_aabb_collision(veh_corners, obs_center, obs_half_w, obs_half_h):
            return True
        
        # Check trailer collision
        if check_obb_aabb_collision(trl_corners, obs_center, obs_half_w, obs_half_h):
            return True
    
    return False

def check_trajectory_collision(states, params, obstacle_list):
    """
    Check if any state in a trajectory collides with obstacles.
    
    Args:
        states: State trajectory array of shape (num_state, horizon+1)
        params: Vehicle parameters dict
        obstacle_list: List of obstacles
        
    Returns:
        True if any collision detected (should use obstacle-aware MPC)
    """
    if len(obstacle_list) == 0:
        return False
    
    num_steps = states.shape[1]
    
    for k in range(num_steps):
        state_k = states[:, k]
        if check_state_collision(state_k, params, obstacle_list):
            return True
    
    return False

if __name__ == "__main__":
    dt_to = 0.1
    dt = 0.05
    horizon = 50
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
    # Create both controllers - switch based on collision detection in horizon
    if USE_SWITCH_MPC:
        controller_obs = MPCTrackingControlObs(model, params,
                                    Q, R, 
                                    state_bound, input_bound, obstacle_list=obstacle_list)
        controller_no_obs = MPCTrackingControl(model, params,
                                    Q, R, 
                                    state_bound, input_bound)
    elif USE_OBS_MPC:
        controller_obs = MPCTrackingControlObs(model, params,
                                    Q, R, 
                                    state_bound, input_bound, obstacle_list=obstacle_list)
        controller_no_obs = MPCTrackingControlObs(model, params,
                                    Q, R, 
                                    state_bound, input_bound, obstacle_list=obstacle_list)
    else:
        controller_obs = MPCTrackingControl(model, params,
                                    Q, R, 
                                    state_bound, input_bound)
        controller_no_obs = MPCTrackingControl(model, params,
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
    
    print("Controller switching: Using obstacle-aware MPC when collision detected in horizon")
    
    # Store previous MPC prediction for collision checking (None on first iteration)
    prev_mpc_prediction = None
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
        
        # Check if MPC prediction collides with obstacles in the horizon
        # Use previous MPC prediction if available, otherwise fall back to reference trajectory
        traj_to_check = prev_mpc_prediction if prev_mpc_prediction is not None else ref_state_traj_
        needs_obstacle_avoidance = check_trajectory_collision(traj_to_check, params, obstacle_list)
        
        if needs_obstacle_avoidance:
            controller = controller_obs
            print("Using obstacle-aware MPC")
        else:
            controller = controller_no_obs
        
        # Create noisy state measurement for MPC (simulates sensor uncertainty)
        if ENABLE_DISTURBANCES:
            measurement_noise = generate_measurement_noise(DISTURBANCE_PARAMS)
            state_measured = state + measurement_noise
        else:
            state_measured = state
        
        t_start = time.perf_counter()
        states, inputs = controller.solve(state_measured, ref_state_traj_, ref_input_traj_)
        t_end = time.perf_counter()
        solve_times.append(t_end - t_start)
        # Store MPC prediction for next iteration's collision check
        prev_mpc_prediction = np.array(states)     
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
    
    # Final state metrics
    goal_state_final = ref_state_traj[:, -1]
    distance_error = np.sqrt((state[0] - goal_state_final[0])**2 + (state[1] - goal_state_final[1])**2)
    heading_error = state[2] - goal_state_final[2]
    hitch_angle_error = state[3] - goal_state_final[3]
    # Normalize angles to [-pi, pi]
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
    hitch_angle_error = (hitch_angle_error + np.pi) % (2 * np.pi) - np.pi
    
    print("FINAL STATE METRICS")
    print(f"  Distance error:        {distance_error:.4f} m")
    print(f"  Tractor heading error: {np.degrees(heading_error):.4f} deg ({heading_error:.4f} rad)")
    print(f"  Hitch angle error:     {np.degrees(hitch_angle_error):.4f} deg ({hitch_angle_error:.4f} rad)")
    
    plt.show()
    