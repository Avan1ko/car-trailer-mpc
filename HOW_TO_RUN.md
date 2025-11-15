# How to Run the Unity Simulation

This guide explains how to run the truck-trailer motion planning simulation in Unity.

## Prerequisites

1. **Unity Editor** - Version 2021.3.9f1 (or compatible)
   - Download from: https://unity.com/
   - The project uses Unity 2021.3.9f1 as specified in `ProjectSettings/ProjectVersion.txt`

2. **Python 3** with the following packages:
   - `numpy`
   - `casadi` (for optimization)
   - `matplotlib` (for visualization)
   - `scipy` (may be required by casadi)

   Install Python dependencies:
   ```bash
   pip install numpy casadi matplotlib scipy
   ```

## Running the Simulation

### Step 1: Open the Project in Unity

1. Launch Unity Hub
2. Click "Add" and select the project folder: `BEP-Motion-planning-for-Truck-Trailers`
3. Unity will detect the project and you can open it
4. Wait for Unity to import all assets (this may take a few minutes on first launch)

### Step 2: Open the Main Scene

The project has two main scenes:
- **`Assets/self-driving.unity`** - Main self-driving vehicle scene (set as default in build settings)
- **`Assets/self-driving-trailer-parking.unity`** - Truck-trailer parking scenario

**To open a scene:**
1. In Unity, go to `File > Open Scene`
2. Navigate to `Assets/` and select either:
   - `self-driving-trailer-parking.unity` (recommended for truck-trailer simulation)
   - `self-driving.unity` (general self-driving vehicle scene)

### Step 3: Run the Simulation in Unity

1. Click the **Play** button (▶) in the Unity Editor
2. The simulation will start and you should see:
   - The warehouse/parking environment
   - The truck and trailer
   - Obstacles in the scene

### Step 4: Generate a Path

1. **Set Start Position**: Click in the scene to place the truck at the starting position
2. **Set Goal Position**: Click again to set the target/goal position
3. The Hybrid A* pathfinding algorithm will automatically:
   - Find a path from start to goal
   - Generate `initialize.json` and `obstacles.json` files in the project root directory
   - These files contain the path data and obstacle information

### Step 5: Run Python Trajectory Optimization (Optional)

After Unity generates the path, you can optimize it using Python:

1. **Navigate to PythonParts directory:**
   ```bash
   cd PythonParts
   ```

2. **Run trajectory optimization:**
   ```bash
   python trajectory_animation.py
   ```
   
   This will:
   - Read `initialize.json` and `obstacles.json` from the project root
   - Optimize the trajectory to make it smoother and collision-free
   - Generate optimized trajectory data

3. **Run MPC simulation (optional):**
   ```bash
   python simulation.py
   ```
   
   This simulates the Model Predictive Controller following the optimized trajectory.

## Workflow Summary

The typical workflow is:

1. **Unity** → Generates initial path using Hybrid A* → Writes `initialize.json` and `obstacles.json`
2. **Python** → Reads JSON files → Optimizes trajectory → (Optional: writes back optimized data)
3. **Unity** → Uses the optimized trajectory to control the truck-trailer

## Important Notes

- The JSON files (`initialize.json`, `obstacles.json`, `states.json`) are written to the **project root directory** (same level as `Assets/` folder)
- Python scripts in `PythonParts/` read from the project root, so make sure you're running Python scripts from the correct directory or adjust paths accordingly
- The simulation uses file-based communication between Unity and Python (JSON files)

## Troubleshooting

- **Unity won't open**: Make sure you have Unity 2021.3.9f1 or a compatible version
- **Python import errors**: Install missing packages with `pip install <package-name>`
- **JSON files not found**: Make sure Unity has generated the path first (click Play and set start/goal positions)
- **Scene not loading**: Check that all assets are imported properly (Unity will show progress in the bottom-right)

## Additional Scenes

The project includes several test scenes in `Assets/Test scenes/`:
- `benchmark-hybrid-A.unity` - Pathfinding benchmark
- `smooth-path.unity` - Path smoothing tests
- `follow-path.unity` - Path following tests
- And more...

You can explore these scenes to test individual components of the system.

