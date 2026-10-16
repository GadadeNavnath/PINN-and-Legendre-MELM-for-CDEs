### Problem 2: Two-Dimensional Steady Convection–Diffusion Problem (Continuous Boundary Conditions)

### 1. Problem Description

Consider the steady two-dimensional convection–diffusion equation

$$
a_x\frac{\partial \phi}{\partial x}
+a_y\frac{\partial \phi}{\partial y}
-k\left(
\frac{\partial^2\phi}{\partial x^2}
+\frac{\partial^2\phi}{\partial y^2}
\right)=5,
\qquad (x,y)\in\Omega,
$$

where

$$
\Omega=\left(-\frac12,\frac12\right)
\times\left(-\frac12,\frac12\right).
$$

The equation is subject to homogeneous Dirichlet boundary conditions

$$
\phi(x,y)=0,\qquad (x,y)\in\partial\Omega.
$$

The problem parameters are:

- Convection coefficients: $$a_x=a_y=\frac{1}{\sqrt{2}}$$
- Convection velocity magnitude: $$U=1$$
- Diffusion coefficient: $$k=0.001$$
- Domain length: $$L=1$$

The corresponding Péclet number is

$$
\mathrm{Pe}=\frac{UL}{k}=1000.
$$

### 2. Reference Solution

An exact analytical solution is not available for this problem. Therefore, the FEM solution is used as the reference solution. The FEM solution is included in `P2_PINN.csv` along with the PINN results.

### 2. Files and Results

| File | Description |
|---|---|
| `P2.py` | Python implementation of Legendre-MELM. |
| `P2_MELM.csv` | MELM numerical results used by `Plots.py`. |
| `P2_PINN.csv` | PINN numerical results and FEM reference solution; required to run `P2.py` and `Plots.py`. |
| `P2_bnd_com.pdf` | Boundary results comparison plot. |
| `P2_contour.pdf` | Contour plot of the solution. |
| `P2_error.pdf` | Error plot. |
| `P2_int_com.pdf` | Interior results comparison plot. |
| `P2_surface.pdf` | Surface plot of the solution. |
| `Plots.py` | Python script for generating plots using `P2_MELM.csv` and `P2_PINN.csv`. |

### 3. How to Run

Run the scripts from the directory containing the required files, in the following order:

```bash
python P2.py
python Plots.py
```

Ensure that `P2_PINN.csv` is available before running `P2.py`, and that both `P2_MELM.csv` and `P2_PINN.csv` are available before running `Plots.py`.


