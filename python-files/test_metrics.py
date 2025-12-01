import numpy as np
import casadi as ca

def _obstacle_Hrep(obstacle):
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
    b += A @ center  # Shift center of half-space representation
    return A, b

def evaluate_sx(sx_expr, *inputs):
    """
    Numerically evaluate a CasADi SX or DM expression and return a NumPy array.
    *inputs are all required numeric arguments to compute the expression.
    """
    # Infer input symbols from the graph by scanning
    syms = []
    for arg in inputs:
        if isinstance(arg, (ca.SX, ca.MX, ca.DM)):
            syms.append(arg)

    # Build a function mapping inferred symbols to sx_expr
    fun = ca.Function("eval_sx", syms, [sx_expr])

    # Convert inputs to raw numpy/float if needed
    fun_inputs = [a.full() if isinstance(a, ca.DM) else a for a in inputs]
    out = fun(*fun_inputs)[0]
    return out.full() if isinstance(out, ca.DM) else np.array(out.full()).squeeze()



def _collision_constraints(model, states, obstacle_list):

    # Convert states to NumPy immediately
    # (Handles SX or DM input)
    states_np = ca.DM(states).full()  # shape should be (4, N)
    N = states_np.shape[1]

    collision_count = 0
    info = []

    for k in range(N):

        # Extract numeric state values
        x_rear =  states_np[0, k]
        y_rear =  states_np[1, k]
        heading = states_np[2, k]
        hitch =   states_np[3, k]

        # Compute centers numerically — convert if needed
        xv_c, yv_c = model.get_vehicle_center(x_rear, y_rear, heading)
        pv_center = np.array([xv_c, yv_c]).reshape(2,1)

        xt_c, yt_c = model.get_trailer_center(x_rear, y_rear, heading, hitch)
        pt_center = np.array([xt_c, yt_c]).reshape(2,1)

        # Rotation matrices numerically (clockwise)
        Rv = np.array([
            [np.cos(heading),  np.sin(heading)],
            [-np.sin(heading), np.cos(heading)]
        ])
        Rt = np.array([
            [np.cos(heading + hitch),  np.sin(heading + hitch)],
            [-np.sin(heading + hitch), np.cos(heading + hitch)]
        ])

        for obs in obstacle_list:

            Ao, bo = _obstacle_Hrep(obs)

            # Convert Ao, bo to numpy
            Ao_np = ca.DM(Ao).full()  # (4,2)
            bo_np = ca.DM(bo).full()  # (4,1)

            # Build combined matrices numerically
            A_vehicle_obs = np.vstack([
                Ao_np,
                (Rv @ model.get_vehicle_Hrep()[0].T).T
            ])

            A_trailer_obs = np.vstack([
                Ao_np,
                (Rt @ model.get_trailer_Hrep()[0].T).T
            ])

            b_vehicle = model.get_vehicle_Hrep()[1] + model.get_vehicle_Hrep()[0] @ pv_center
            b_trailer = model.get_trailer_Hrep()[1] + model.get_trailer_Hrep()[0] @ pt_center

            b_vehicle_obs = np.vstack([
                bo_np,
                b_vehicle
            ])
            b_trailer_obs = np.vstack([
                bo_np,
                b_trailer
            ])

            # Rank test using NumPy

            AB_vehicle = np.hstack([A_vehicle_obs, b_vehicle_obs])
            AB_trailer = np.hstack([A_trailer_obs, b_trailer_obs])
            rankA_vehicle  = np.linalg.matrix_rank(A_vehicle_obs)
            rankAB_vehicle = np.linalg.matrix_rank(AB_vehicle)
            rankA_trailer = np.linalg.matrix_rank(A_trailer_obs)
            rankAB_trailer = np.linalg.matrix_rank(AB_trailer) 

            if (rankA_vehicle == rankAB_vehicle and rankA_vehicle <= A_vehicle_obs.shape[0]) or (rankA_trailer == rankAB_trailer and rankA_trailer <= A_trailer_obs.shape[0]):
                collision_count += 1
                info.append({
                    "position": [x_rear, y_rear],
                    "time_step": k,
                    "obstacle": obs
                })

    return collision_count, info
