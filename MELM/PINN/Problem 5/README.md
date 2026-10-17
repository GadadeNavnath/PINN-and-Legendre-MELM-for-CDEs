### Problem 5: Two-Dimensional Unsteady Convection–Diffusion Problem

The mathematical formulation, exact solution, boundary conditions, and problem parameters are provided in the `README.md` file under `MELM/MELM/Problem_5/`.

### 1. Continuous-Time (CT) Formulation

#### Files and Results

| File | Description |
|---|---|
| `data/` | Directory containing CSV files required to run `Plots.py`. |
| `Plots.py` | Python script for generating CT plots using the CSV files in `data/`. |
| `L2_error.py` | Python script for computing the $L^2$ error. |
| `P5_CT_PINN.pdf` | Solution plot for CT PINN. |

#### How to Run

Run the following command from the `CT/` directory:

```bash
python Plots.py
```



### 2. Discrete-Time (DT) Formulation

#### Files and Results

| File | Description |
|---|---|
| `data.csv` | CSV file containing numerical data required for generating DT PINN plots. |
| `Plots1.py` | Python script for generating DT PINN plots. |
| `Plots2.py` | Python script for generating additional DT PINN plots. |
| `P5_DT_PINN_solution.pdf` | Solution plot for DT PINN. |
| `P5_DT_PINN_error_vs_time.pdf` | Error versus time plot for DT PINN. |

#### How to Run

Run the following commands from the `DT/` directory:

```bash
python Plots1.py
python Plots2.py
```
