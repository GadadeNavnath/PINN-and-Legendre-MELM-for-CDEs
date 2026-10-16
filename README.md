### PINN and Legendre-MELM for CDEs

PINN and Legendre-Based Modified Extreme Learning Machine (MELM) Methods for Solving Convection–Diffusion Equations.

This repository contains reproducibility materials for the manuscript:

**Title:** “ A Legendre-Based Modified Extreme Learning Machine for Convection–Diffusion Equations: Steady-State, Continuous-Time, and Discrete-Time Formulations”

---

This repository provides Python implementations for solving convection–diffusion equations (CDEs). The code is organized into five benchmark problems covering steady and unsteady CDEs, with both continuous-time and discrete-time formulations.

---

### Requirements

#### PINN implementation requirements:

Pls update here 


#### MELM implementation Requirements:

The code is implemented in Python and requires the following packages:

- Python 3.12 or later
- NumPy
- Matplotlib
- Pandas
- SciPy

Install the required packages using:

pip install numpy matplotlib pandas scipy

## Problem Overview

| Problem | Dimension | Type | Formulation |
|---|---|---|---|
| Problem 1 | 1D | Steady | Steady |
| Problem 2 | 2D | Steady | Steady |
| Problem 3 | 2D | Steady | Steady |
| Problem 4 | 2D (CT), 1D (DT) | Unsteady | Continuous-time (CT) and discrete-time (DT) |
| Problem 5 | 3D (CT), 2D (DT) | Unsteady | Continuous-time (CT) and discrete-time (DT) |

### Repository Structure

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
    │   │   └── [PINN CSV files, numerical results, and plots]
    │   │
    │   ├── Problem_2/
    │   │   ├── [MELM code and results]
    │   │   └── [PINN CSV files, numerical results, and plots]
    │   │
    │   ├── Problem_3/
    │   │   ├── [MELM code and results]
    │   │   └── [PINN CSV files, numerical results, and plots]
    │   │
    │   ├── Problem_4/
    │   │   ├── CT/
    │   │   │   ├── [MELM Python scripts]
    │   │   │   ├── [CSV files and numerical results]
    │   │   │   └── [Plots]
    │   │   └── DT/
    │   │       ├── [MELM Python scripts]
    │   │       ├── [CSV files and numerical results]
    │   │       └── [Plots]
    │   │
    │   └── Problem_5/
    │       ├── CT/
    │       │   ├── [MELM Python scripts]
    │       │   ├── [CSV files and numerical results]
    │       │   └── [Plots]
    │       └── DT/
    │           ├── [MELM Python scripts]
    │           ├── [CSV files and numerical results]
    │           └── [Plots]
    │
    └── PINN/
        ├── Problem_4/
        │   ├── CT/
        │   │   └── [CSV files, numerical results, and plots]
        │   └── DT/
        │       └── [CSV files, numerical results, and plots]
        │
        └── Problem_5/
            ├── CT/
            │   └── [CSV files, numerical results, and plots]
            └── DT/
                └── [CSV files, numerical results, and plots]
```

### Directory Descriptions

- **`PINN/`**: Contains the Python scripts implementing the Physics-Informed Neural Network (PINN) method.
- **`MELM/MELM/`**: Contains the MELM implementations and numerical results for all five benchmark problems. For Problems 1–3, the corresponding PINN CSV files, numerical results, and plots are also included in the respective problem folders.
- **`MELM/MELM/Problem_4/` and `MELM/MELM/Problem_5/`**: Each contains `CT/` and `DT/` subfolders with the MELM Python scripts, CSV files, numerical results, and plots for the respective formulations.
- **`MELM/PINN/`**: Contains the PINN numerical results for Problems 4 and 5, organized into `CT/` and `DT/` subfolders, with separate CSV files and plots for each formulation.

 ### Reproducibility

The code and associated numerical results are provided to support reproducibility of the computational experiments presented in the accompanying manuscript.

For comparisons between PINN and MELM, refer to the corresponding problem directories and their numerical results. The CT and DT subfolders distinguish the continuous-time and discrete-time formulations for Problems 4 and 5.

