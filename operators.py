import random
import numpy as np
from config import (N_WAYPOINTS, Z_MIN, Z_MAX, TARGET_POS,
                    CROSSOVER_RATE, MUTATION_RATE, TOURNAMENT_SIZE)


def tournament_select(population: list, fitnesses: list) -> list:
    """Select one individual via tournament selection."""
    competitors = random.sample(range(len(population)), TOURNAMENT_SIZE)
    winner      = min(competitors, key=lambda i: fitnesses[i])
    return population[winner][:]


def crossover(parent1: list, parent2: list) -> tuple:
    """Single-point crossover at a waypoint boundary."""
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]
    # gene index must be a multiple of 3 (one full waypoint)
    point  = random.randint(1, N_WAYPOINTS - 1) * 3
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(chromosome: list) -> list:
    """Gaussian perturbation of randomly selected waypoints."""
    chrom = chromosome[:]
    for wp_idx in range(N_WAYPOINTS):
        if random.random() < MUTATION_RATE:
            base = wp_idx * 3
            chrom[base]     += np.random.normal(0, 5.0)   # x
            chrom[base + 1] += np.random.normal(0, 5.0)   # y
            chrom[base + 2]  = np.clip(
                chrom[base + 2] + np.random.normal(0, 5.0),
                Z_MIN, Z_MAX
            )                                               # z (clamped)
    return chrom


def repair(chromosome: list) -> list:
    """Nudge the final waypoint toward the target (soft constraint enforcement)."""
    chrom = chromosome[:]
    last  = (N_WAYPOINTS - 1) * 3
    for d in range(3):
        chrom[last + d] = 0.7 * chrom[last + d] + 0.3 * TARGET_POS[d]
    return chrom
