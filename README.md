# 🚀 Genetic Algorithm for Ballistic Missile Trajectory Optimization

> Evolutionary path planning for ballistic missiles navigating hostile war-zone environments with enemy defense zones, altitude constraints, and G-force limits.

---

## 📖 Description

This project applies a **Genetic Algorithm (GA)** to solve the multi-objective problem of planning an optimal 3D ballistic missile trajectory. The missile must travel from a launch position to an enemy target while minimizing fuel consumption and flight instability, and avoiding enemy defense zones — all under realistic physical constraints.

The GA evolves a population of candidate trajectories (chromosomes) over multiple generations using tournament selection, single-point crossover, Gaussian mutation, and elitism to converge on the best feasible path.

---

## 🗂️ Project Structure

```
├── config.py          # All problem & GA hyperparameters
├── trajectory.py      # Chromosome encoding, decoding, random initialization
├── fitness.py         # Fitness function: fuel cost + turning penalty + constraint penalties
├── operations.py      # GA operators: selection, crossover, mutation, repair
└── main.py            # Entry point — runs the GA and visualizes results
```

---

## ⚙️ How It Works

### Chromosome Representation
Each chromosome encodes **6 intermediate 3D waypoints** (18 floats) between the fixed launch and target positions. The full trajectory is reconstructed as:

```
LAUNCH → WP1 → WP2 → ... → WP6 → TARGET
```

### Fitness Function
The fitness is minimized and combines three components:

| Component | Description |
|---|---|
| **Fuel Cost** | Path length weighted by altitude-dependent aerodynamic drag |
| **Turning Cost** | Penalizes sharp direction changes for flight stability |
| **Constraint Penalties** | Enemy zone intrusion, altitude bounds, G-force limit, target miss distance |

### GA Pipeline
1. **Initialization** — Random population biased toward the straight-line path to the target
2. **Selection** — Tournament selection (size 5)
3. **Crossover** — Single-point crossover at waypoint boundaries (rate: 85%)
4. **Mutation** — Gaussian perturbation of waypoint coordinates (rate: 12%)
5. **Repair** — Soft nudge of the final waypoint toward the target
6. **Elitism** — Top 10% of individuals preserved unchanged each generation

---

## 🗺️ Problem Configuration

| Parameter | Value |
|---|---|
| Launch Position | (0, 0, 0) |
| Target Position | (100, 80, 0) |
| Intermediate Waypoints | 6 |
| Altitude Range | 2 – 50 units |
| Max G-Force | 30 units/s² |
| Hit Tolerance (ε) | 3.0 units |
| Enemy Defense Zones | 4 spherical zones |

### Enemy Defense Zones
| Zone | Center (x, y, z) | Radius |
|---|---|---|
| 1 | (30, 20, 10) | 12 |
| 2 | (60, 50, 8) | 10 |
| 3 | (50, 30, 15) | 8 |
| 4 | (80, 60, 5) | 9 |

---

## 🧬 GA Hyperparameters

| Parameter | Value |
|---|---|
| Population Size | 200 |
| Max Generations | 500 |
| Crossover Rate | 0.85 |
| Mutation Rate | 0.12 |
| Tournament Size | 5 |
| Elitism Rate | 10% |
| Turning Penalty Weight (α) | 2.0 |
| Constraint Penalty Weight (λ) | 50.0 |

---

## 🚀 Getting Started

### Requirements
```bash
pip install numpy matplotlib
```

### Run
```bash
python main.py
```

The GA will print generation-by-generation progress, display convergence plots, and output the best trajectory found along with its fitness breakdown.

---

## 📊 Output

- **Convergence curve** — Best and average fitness over generations
- **3D trajectory plot** — Optimized missile path with enemy zones visualized
- **Fitness breakdown** — Fuel cost, turning cost, and total penalty for the best solution
- **JSON export** — Best chromosome and fitness history saved to file

---

## 📐 Penalty Scheme

| Violation | Penalty |
|---|---|
| Enemy zone intrusion | 100 × penetration depth (per waypoint) |
| Altitude below Z_MIN | 50 × violation magnitude |
| Altitude above Z_MAX | 50 × violation magnitude |
| G-force exceeded | 20 × excess acceleration |
| Target miss > ε | 200 × miss distance |

---

## 📝 License

This project is for academic and research purposes only.
