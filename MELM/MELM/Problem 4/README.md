### Problem 4: One-Dimensional Unsteady Convection–Diffusion Problem

### 1. Problem Description

Consider the one-dimensional unsteady convection–diffusion equation

$$
\frac{\partial \phi}{\partial t}
+a\frac{\partial \phi}{\partial x}
-k\frac{\partial^2\phi}{\partial x^2}=0,
\qquad (x,t)\in(0,L)\times(0,T].
$$

The exact solution is given by

$$
\phi(x,t)=
\frac{1}{\sqrt{4t+1}}
\exp\left(
-\frac{(x-1-at)^2}{k(4t+1)}
\right).
$$

The initial condition and Dirichlet boundary conditions are obtained directly from the exact solution.

The problem parameters are:

- Convection velocity: $$a=1$$
- Diffusion coefficient: $$k=0.1$$
- Domain length: $$L=9$$
- Final time: $$T=5$$
- Convection velocity magnitude: $$U=1$$

The corresponding Péclet number is

$$
\mathrm{Pe}=\frac{UL}{k}=90.
$$

## 2. Continuous-Time (CT) Formulation

### Files and Results

| File | Description |
|---|---|
| `P4.ipynb` | Jupyter notebook implementing CT Legendre-MELM. |
| `P4_CT_MELM.csv` | CT MELM numerical results. |
| `P4_CT_MELM_abs_error.pdf` | Absolute error plot for CT MELM. |
| `P4_CT_MELM_cont.pdf` | Contour plot of the CT MELM solution. |
| `P4_CT_MELM_sol_profile.pdf` | Solution profile plot for CT MELM. |
| `P4_exact.pdf` | Exact solution plot. |
| `Plots.ipynb` | Jupyter notebook for generating CT plots. |

### How to Run

Open and execute `P4.ipynb` in the `CT/` directory. To generate the plots, execute `Plots.ipynb` if required.

## 3. Discrete-Time (DT) Formulation

### Files and Results

| File | Description |
|---|---|
| `P4.ipynb` | Jupyter notebook implementing DT Legendre-MELM. |
| `P4_DT_MELM.csv` | DT MELM numerical results. |
| `P4_DT_MELM_absolute_error.pdf` | Absolute error plot for DT MELM. |
| `P4_DT_MELM_error_vs_time.pdf` | Error versus time plot for DT MELM. |
| `P4_DT_MELM_solution.pdf` | Solution plot for DT MELM. |

### How to Run

Open and execute `P4.ipynb` in the `DT/` directory.


