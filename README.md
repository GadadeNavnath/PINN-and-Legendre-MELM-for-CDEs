# PINN and Legendre-MELM for CDEs

PINN and Legendre-Based Modified Extreme Learning Machine (MELM) Methods for Solving Convection–Diffusion Equations.

This repository contains reproducibility materials for the manuscript:

**Title:** “ A Legendre-Based Modified Extreme Learning Machine for Convection–Diffusion Equations: Steady-State, Continuous-Time, and Discrete-Time Formulations”

---

This repository provides Python implementations for solving convection–diffusion equations (CDEs). The code is organized into five benchmark problems covering steady and unsteady CDEs, with both continuous-time and discrete-time formulations.

---

### Requirements

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
| Problem 4 | 2D Problem | Unsteady (CT and DT) |
| Problem 5 | 3D Problem | Unsteady (CT and DT) |

## Repository Structure

The repository is organized into separate directories for the PINN source code, MELM implementations, and numerical results obtained using both methods. Problems 4 and 5 are further divided into continuous-time (CT) and discrete-time (DT) formulations, with separate CSV files and numerical results for each formulation.

```text
PINN-and-Legendre-MELM-for-CDEs/
├── PINN/
│   └── [PINN Python scripts]
│
└── MELM/
    ├── MELM/
    │   ├── Problem_1/
    │   │   ├── [MELM code and results]
    │   │   └── [PINN CSV files, results, and plots]
    │   ├── Problem_2/
    │   │   ├── [MELM code and results]
    │   │   └── [PINN CSV files, results, and plots]
    │   ├── Problem_3/
    │   │   ├── [MELM code and results]
    │   │   └── [PINN CSV files, results, and plots]
    │   ├── Problem_4/
    │   │   ├── CT/
    │   │   │   └── [MELM CT CSV files and results]
    │   │   └── DT/
    │   │       └── [MELM DT CSV files and results]
    │   └── Problem_5/
    │       ├── CT/
    │       │   └── [MELM CT CSV files and results]
    │       └── DT/
    │           └── [MELM DT CSV files and results]
    │
    └── PINN/
        ├── Problem_4/
        │   ├── CT/
        │   │   └── [PINN CT CSV files and results]
        │   └── DT/
        │       └── [PINN DT CSV files and results]
        └── Problem_5/
            ├── CT/
            │   └── [PINN CT CSV files and results]
            └── DT/
                └── [PINN DT CSV files and results]
```

### Directory Descriptions

- **`PINN/`**: Contains the Python scripts implementing the Physics-Informed Neural Network (PINN) method.
- **`MELM/MELM/`**: Contains the MELM implementations and numerical results for all five benchmark problems. For Problems 1–3, the corresponding PINN CSV files, numerical results, and plots are included in the respective problem folders.
- **`MELM/MELM/Problem_4/` and `MELM/MELM/Problem_5/`**: Each contains separate `CT/` and `DT/` subfolders with the corresponding MELM CSV files and numerical results for the continuous-time and discrete-time formulations.
- **`MELM/PINN/`**: Contains the PINN numerical results for Problems 4 and 5. Each problem is divided into `CT/` and `DT/` subfolders, containing separate CSV files and numerical results, including plots, for each formulation.

This organization separates the PINN source code from its numerical results and keeps the results for each method, benchmark problem, and time formulation clearly organized for reproducibility and comparison.
