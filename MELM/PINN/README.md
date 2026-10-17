### Problem 4: One-Dimensional Unsteady Convection–Diffusion Problem

The mathematical formulation, exact solution, boundary conditions, and problem parameters are provided in the `README.md` file under `MELM/MELM/Problem_4/`.

### 1. Continuous-Time (CT) Formulation

#### Files and Results

| File | Description |
|---|---|
| `data/` | Directory containing CSV files required to run `Plots.py`. |
| `Plots.py` | Python script for generating CT plots using the CSV files in `data/`. |
| `P4_CT_PINN_abs_error.pdf` | Absolute error plot for CT PINN. |
| `P4_CT_PINN_cont.pdf` | Contour plot of the CT PINN solution. |
| `P4_CT_PINN_sol_profile.pdf` | Solution profile plot for CT PINN. |

#### How to Run

Run the following command from the `CT/` directory:

```bash
python Plots.py
```

### 2. Discrete-Time (DT) Formulation

#### Files and Results

| File | Description |
|---|---|
| `data/` | Directory containing CSV files required to run `Plots.py`. |
| `Plots.py` | Python script for generating DT plots using the CSV files in `data/`. |
| `P4_DT_PINN_absolute_error.pdf` | Absolute error plot for DT PINN. |
| `P4_DT_PINN_error_vs_time.pdf` | Error versus time plot for DT PINN. |
| `P4_DT_PINN_solution.pdf` | Solution plot for DT PINN. |

#### How to Run

Run the following command from the `DT/` directory:

```bash
python Plots.py
```
