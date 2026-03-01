import numpy as np
from config import LAUNCH_POS, TARGET_POS, N_WAYPOINTS, Z_MIN, Z_MAX


def encode(waypoints: np.ndarray) -> list:
    """Flatten (N,3) array → 1-D list for GA operations."""
    return waypoints.flatten().tolist()


def decode(chromosome: list) -> np.ndarray:
    """Reshape flat list → (N,3) waypoint array."""
    return np.array(chromosome).reshape(-1, 3)


def random_chromosome() -> list:
    """
    Create a random trajectory biased toward the target.
    Waypoints are sampled along the straight-line path with noise.
    """
    pts = []
    for i in range(1, N_WAYPOINTS + 1):
        t = i / (N_WAYPOINTS + 1)
        base = LAUNCH_POS + t * (TARGET_POS - LAUNCH_POS)
        noise_xy = np.random.uniform(-20, 20, 2)
        noise_z  = np.random.uniform(Z_MIN, Z_MAX)
        pts.append([base[0] + noise_xy[0],
                    base[1] + noise_xy[1],
                    noise_z])
    return encode(np.array(pts))
