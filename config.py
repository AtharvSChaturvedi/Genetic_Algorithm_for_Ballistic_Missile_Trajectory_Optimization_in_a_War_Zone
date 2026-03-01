import numpy as np

# PROBLEM CONFIGURATION
LAUNCH_POS   = np.array([0.0,   0.0,   0.0])    # missile launch position
TARGET_POS   = np.array([100.0, 80.0, 0.0])     # enemy depot coordinates
N_WAYPOINTS  = 6                                  # intermediate 3D waypoints
HIT_TOL      = 3.0                               # acceptable miss distance (ε)

# Flight envelope
Z_MIN, Z_MAX = 2.0, 50.0

# G-force (acceleration) limit
A_MAX = 30.0   # units/s²

# Enemy defense zones: list of (center_x, center_y, center_z, radius)
ENEMY_ZONES = [
    (30.0, 20.0, 10.0, 12.0),
    (60.0, 50.0,  8.0, 10.0),
    (50.0, 30.0, 15.0,  8.0),
    (80.0, 60.0,  5.0,  9.0),
]

# GA HYPER-PARAMETERS
POP_SIZE        = 200      # candidate trajectories per generation
N_GENERATIONS   = 500      # maximum evolution iterations
CROSSOVER_RATE  = 0.85     # probability of crossover
MUTATION_RATE   = 0.12     # probability of waypoint mutation
TOURNAMENT_SIZE = 5        # selection tournament size
ELITISM_RATE    = 0.10     # top % preserved unchanged
ALPHA           = 2.0      # penalty weight for sharp turns
LAMBDA          = 50.0     # penalty weight for constraint violations
