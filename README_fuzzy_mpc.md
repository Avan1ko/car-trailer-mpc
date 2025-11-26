# Fuzzy weight scaling for the MPC

This repo includes a light fuzzy layer on top of the MPC: the cost weights are scaled at every solve using simple rules based on hitch angle and direction of travel. When the trailer starts to fold or when reversing, the controller increases penalties on hitch/steering states and steering rate to discourage jackknifing without changing constraints or the solver. The fuzzy controller lives separately from the baseline MPC.

## Rules used (python-files/mpc_control.py)
- Monitor current hitch angle `psi` and velocity `v` (and the current reference velocity).
- Normalize `|psi|` against a soft threshold of 0.35 rad (~20°).
- If reversing (`v` or reference `v` < -0.1 m/s), boost the response.
- Scale diagonals of Q/R each solve:
  - Hitch (`psi`), steering angle (`phi`), and heading (`theta`) get larger Q weights as `|psi|` grows (stronger when reversing).
  - Steering rate input gets a larger R weight as `|psi|` grows (stronger when reversing).

## How to run
Use the dedicated fuzzy sim so the baseline stays unchanged:
```
cd /Users/runingguan/car-trailer-mpc
MPLCONFIGDIR=.mplconfig python python-files/simulation_fuzzy.py
```
Defaults: disturbances off, horizon 40, T_sim 30s. The sim reuses the last control if the solver hiccups and eventually zeros/stops after repeated failures.

## Tuning knobs
- Soft hitch threshold: set via `hitch_soft` in `_compute_fuzzy_weights` (python-files/mpc_control_fuzzy.py).
- Gain multipliers: adjust `hitch_gain`, `steer_gain`, `steer_rate_gain` factors and reversing multipliers to make the controller more or less conservative when the trailer folds or when backing up. Current gains are conservative and clipped to [1, 3.5].
- Warm start: the solver shifts the last optimal trajectory for the next call. If solves fail, it retries with nominal weights, then the sim holds/zeros control after repeated failures.
- Disturbances: set `ENABLE_DISTURBANCES` in simulation_fuzzy.py once the nominal run is stable.

## NMPC variant
If you want the full nonlinear MPC instead of fuzzy weight scaling, use:
```
cd /Users/runingguan/car-trailer-mpc
MPLCONFIGDIR=.mplconfig python python-files/simulation_nmpc.py
```
This uses `mpc_control_nmpc.py`, a shorter horizon (30), relaxed Ipopt tolerances, and warm-starting. Disturbances are off by default; re-enable them in `simulation_nmpc.py` after you confirm stability.
