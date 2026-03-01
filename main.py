import random
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from config import (LAUNCH_POS, TARGET_POS, N_WAYPOINTS, HIT_TOL,
                    ENEMY_ZONES, POP_SIZE, N_GENERATIONS, ELITISM_RATE)
from trajectory import random_chromosome, decode
from fitness import fitness, fuel_cost, turning_cost, constraint_penalties
from operators import tournament_select, crossover, mutate, repair


# MAIN GENETIC ALGORITHM
def run_ga():
    print("=" * 60)
    print("  Ballistic Missile Trajectory Optimization – GA")
    print("=" * 60)
    print(f"  Population: {POP_SIZE} | Generations: {N_GENERATIONS}")
    print(f"  Elitism: {ELITISM_RATE * 100:.0f}%")
    print("=" * 60)

    population  = [repair(random_chromosome()) for _ in range(POP_SIZE)]
    n_elite     = max(1, int(ELITISM_RATE * POP_SIZE))

    best_fitness_history = []
    avg_fitness_history  = []
    best_chromosome      = None
    best_fit             = float('inf')

    start_time = time.time()

    for gen in range(N_GENERATIONS):
        fitnesses = [fitness(c) for c in population]

        gen_best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        gen_best_fit = fitnesses[gen_best_idx]
        if gen_best_fit < best_fit:
            best_fit        = gen_best_fit
            best_chromosome = population[gen_best_idx][:]

        best_fitness_history.append(best_fit)
        avg_fitness_history.append(sum(fitnesses) / len(fitnesses))

        if gen % 50 == 0 or gen == N_GENERATIONS - 1:
            elapsed = time.time() - start_time
            print(f"  Gen {gen:4d} | Best Fitness: {best_fit:8.2f} | "
                  f"Avg: {avg_fitness_history[-1]:8.2f} | "
                  f"Time: {elapsed:.1f}s")

        # Convergence check
        if gen >= 100:
            if best_fitness_history[-100] - best_fit < 0.01:
                print(f"\n  Converged at generation {gen} (no improvement in 100 gens)")
                break

        # Elitism
        sorted_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
        elite      = [population[i][:] for i in sorted_idx[:n_elite]]

        # New population
        new_pop = elite[:]
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            c1 = repair(mutate(c1))
            c2 = repair(mutate(c2))
            new_pop.extend([c1, c2])

        population = new_pop[:POP_SIZE]

    elapsed = time.time() - start_time
    print(f"\n  Total evolution time: {elapsed:.2f}s")
    return best_chromosome, best_fit, best_fitness_history, avg_fitness_history


# RESULT REPORT
def report(best_chromosome, best_fit):
    pts       = decode(best_chromosome)
    full_path = np.vstack([LAUNCH_POS, pts, TARGET_POS])
    pen       = constraint_penalties(pts)
    fuel      = fuel_cost(pts)
    turn      = turning_cost(pts)
    miss      = np.linalg.norm(pts[-1] - TARGET_POS)

    print("\n" + "=" * 60)
    print("  OPTIMAL TRAJECTORY REPORT")
    print("=" * 60)
    print(f"  Best Fitness Score    : {best_fit:.4f}")
    print(f"  Fuel Cost             : {fuel:.4f}")
    print(f"  Turning Cost          : {turn:.4f} rad")
    print(f"  Constraint Penalties  : {pen:.4f}")
    print(f"  Final Miss Distance   : {miss:.4f}  (tolerance = {HIT_TOL})")
    print(f"  Target Hit            : {'YES' if miss <= HIT_TOL else 'NO'}")
    print()
    print("  Waypoints (3D coordinates):")
    print(f"    {'Point':<12} {'X':>8} {'Y':>8} {'Z (alt)':>10}")
    print(f"    {'-' * 40}")
    labels = ['Launch'] + [f'WP {i+1}' for i in range(N_WAYPOINTS)] + ['Target']
    for label, pt in zip(labels, full_path):
        print(f"    {label:<12} {pt[0]:>8.2f} {pt[1]:>8.2f} {pt[2]:>10.2f}")
    print("=" * 60)

    # Save results to JSON
    results = {
        "best_fitness": best_fit,
        "fuel_cost": fuel,
        "turning_cost_rad": turn,
        "constraint_penalties": pen,
        "miss_distance": miss,
        "target_hit": bool(miss <= HIT_TOL),
        "waypoints": [
            {"label": label, "x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])}
            for label, pt in zip(labels, full_path)
        ]
    }
    with open("missile_trajectory_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved to missile_trajectory_results.json")


# VISUALIZATION
def visualize(best_chromosome, best_fit, history, avg_history):
    pts       = decode(best_chromosome)
    full_path = np.vstack([LAUNCH_POS, pts, TARGET_POS])

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("Ballistic Missile Trajectory Optimization – Genetic Algorithm",
                 fontsize=13, fontweight='bold')

    # 3D Trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(full_path[:, 0], full_path[:, 1], full_path[:, 2],
             'b-o', linewidth=2, markersize=4, label='Optimal Trajectory')
    ax1.scatter(*LAUNCH_POS, color='green', s=100, zorder=5, label='Launch')
    ax1.scatter(*TARGET_POS, color='red',   s=100, zorder=5, label='Target (Depot)')

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0,     np.pi, 20)
    for (cx, cy, cz, r) in ENEMY_ZONES:
        xs = cx + r * np.outer(np.cos(u), np.sin(v))
        ys = cy + r * np.outer(np.sin(u), np.sin(v))
        zs = cz + r * np.outer(np.ones_like(u), np.cos(v))
        ax1.plot_wireframe(xs, ys, zs, color='red', alpha=0.15, linewidth=0.5)

    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z (Altitude)')
    ax1.set_title('Optimal Trajectory (3D)')
    ax1.legend(loc='upper left', fontsize=7)

    # Convergence Curve
    ax2 = fig.add_subplot(132)
    ax2.plot(history,     label='Best Fitness',    color='blue',   linewidth=1.5)
    ax2.plot(avg_history, label='Average Fitness', color='orange', linewidth=1.0,
             linestyle='--')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Value')
    ax2.set_title('Convergence Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 2D Top-Down View
    ax3 = fig.add_subplot(133)
    ax3.plot(full_path[:, 0], full_path[:, 1], 'b-o',
             linewidth=2, markersize=5, label='Trajectory')
    ax3.scatter(*LAUNCH_POS[:2], color='green', s=120, zorder=5, label='Launch')
    ax3.scatter(*TARGET_POS[:2], color='red',   s=120, zorder=5, label='Target')
    for i, wp in enumerate(pts):
        ax3.annotate(f'WP{i+1}', (wp[0], wp[1]), textcoords='offset points',
                     xytext=(4, 4), fontsize=7, color='blue')

    for (cx, cy, cz, r) in ENEMY_ZONES:
        circle = plt.Circle((cx, cy), r, color='red', alpha=0.2, label='Enemy Zone')
        ax3.add_patch(circle)
    ax3.set_xlabel('X'); ax3.set_ylabel('Y')
    ax3.set_title('Top-Down View (XY Plane)')
    ax3.set_aspect('equal')
    ax3.legend(loc='upper left', fontsize=7)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trajectory_result.png', dpi=150, bbox_inches='tight')
    print("\n  Plot saved to trajectory_result.png")
    plt.show()


# ENTRY POINT
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    best_chrom, best_fit, history, avg_history = run_ga()
    report(best_chrom, best_fit)
    visualize(best_chrom, best_fit, history, avg_history)
