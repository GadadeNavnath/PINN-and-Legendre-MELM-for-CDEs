# Neural-solvers-for-CDEs
FEM, PINN and IELM methods for solving convection–diffusion equations.

This repository contains reproducibility materials for the manuscript:

**Title:** “A Modified Extreme Learning Machine Based on Shifted
Legendre Polynomials for Convection–Diffusion Equations with
Steep Gradients”

**Authors:**

---

This repository provides Python implementations for solving convection–diffusion equations (CDEs). The code is organized into five benchmark problems covering steady and unsteady CDEs, with both continuous-time and discrete-time formulations.

---

### Requirements

#### FEM implementation Requirements:

#### PINN implementation requirements:


#### MELM implementation Requirements:

The code is implemented in Python and requires the following packages:

- Python 3.12 or later
- NumPy
- Matplotlib
- Pandas
- SciPy

Install the required packages using:

pip install numpy matplotlib pandas scipy

### Problem Overview

| Problem   | Type       | Formulation          |
| --------- | ---------- | -------------------- |
| Problem 1 | 1D Problem | Steady               |
| Problem 2 | 2D Problem | Steady               |
| Problem 3 | 2D Problem | Steady               |
| Problem 4 | 1D Problem | Unsteady (CT and DT) |
| Problem 5 | 2D Problem | Unsteady (CT and DT) |

### Repository Structure

The repository is organized into three main folders: **FEM**, **PINN**, and **MELM**. The main `MELM` folder contains separate sub-folders for the MELM and PINN implementations. The `MELM` sub-folder contains code and results for all five benchmark problems, while the `PINN` sub-folder contains results for Problems 4 and 5.

```text
repository/
├── FEM/
├── PINN/
└── MELM/
    ├── MELM/
    │   ├── Problem_1/
    │   ├── Problem_2/
    │   ├── Problem_3/
    │   ├── Problem_4/
    │   └── Problem_5/
    │
    └── PINN/
        ├── Problem_4/
        └── Problem_5/
