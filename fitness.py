import math
import numpy as np
from config import (LAUNCH_POS, TARGET_POS, Z_MIN, Z_MAX,
                    A_MAX, ENEMY_ZONES, ALPHA, LAMBDA, HIT_TOL)
from trajectory import decode


def fuel_cost(pts: np.ndarray) -> float:
    """
    Sum of segment distances weighted by altitude-dependent drag.
    High altitude → lower drag coefficient (Ci decreases).
    """
    full_path = np.vstack([LAUNCH_POS, pts, TARGET_POS])
    total = 0.0
    for i in range(len(full_path) - 1):
        seg_len = np.linalg.norm(full_path[i + 1] - full_path[i])
        avg_z   = (full_path[i][2] + full_path[i + 1][2]) / 2.0
        # Drag: high altitude → less drag (0.5 at z=Z_MAX, 1.5 at z=Z_MIN)
        C_i = 1.5 - (avg_z - Z_MIN) / (Z_MAX - Z_MIN)
        total += C_i * seg_len
    return total


def turning_cost(pts: np.ndarray) -> float:
    """Penalize sharp turns (large direction changes) for flight stability."""
    full_path = np.vstack([LAUNCH_POS, pts, TARGET_POS])
    cost = 0.0
    for i in range(1, len(full_path) - 1):
        v1 = full_path[i]     - full_path[i - 1]
        v2 = full_path[i + 1] - full_path[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angle = math.acos(cos_a)   # radians
            cost += angle
    return cost


def constraint_penalties(pts: np.ndarray) -> float:
    """
    Returns total penalty for constraint violations:
      1. Enemy zone intrusion
      2. Altitude bounds
      3. G-force (acceleration) limit
      4. Target miss distance
    """
    penalty   = 0.0
    full_path = np.vstack([LAUNCH_POS, pts, TARGET_POS])

    for wp in full_path:
        # 1. Enemy zones
        for (cx, cy, cz, r) in ENEMY_ZONES:
            center = np.array([cx, cy, cz])
            dist   = np.linalg.norm(wp - center)
            if dist < r:
                penetration = r - dist
                penalty += 100.0 * penetration

        # 2. Altitude bounds
        if wp[2] < Z_MIN:
            penalty += 50.0 * (Z_MIN - wp[2])
        if wp[2] > Z_MAX:
            penalty += 50.0 * (wp[2] - Z_MAX)

    # 3. G-force (acceleration) – approximate: |Δv/Δt| ≈ change in direction * speed
    dt = 1.0  # unit time step
    for i in range(1, len(full_path) - 1):
        v1  = (full_path[i]     - full_path[i - 1]) / dt
        v2  = (full_path[i + 1] - full_path[i])     / dt
        acc = np.linalg.norm(v2 - v1)
        if acc > A_MAX:
            penalty += 20.0 * (acc - A_MAX)

    # 4. Target accuracy
    miss = np.linalg.norm(pts[-1] - TARGET_POS)
    if miss > HIT_TOL:
        penalty += 200.0 * miss

    return penalty


def fitness(chromosome: list) -> float:
    """Lower is better: fuel + turning penalty + constraint violations."""
    pts = decode(chromosome)
    f   = fuel_cost(pts) + ALPHA * turning_cost(pts)
    pen = constraint_penalties(pts)
    return f + LAMBDA * pen
