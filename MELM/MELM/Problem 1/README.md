### Problem 1: One-Dimensional Steady Convection–Diffusion Problem

### 1. Problem Description

Consider the one-dimensional steady convection–diffusion equation

\[
-k\frac{d^2\phi}{dx^2}+a\frac{d\phi}{dx}=0,
\qquad x\in(0,1),
\]

subject to the Dirichlet boundary conditions

\[
\phi(0)=1,\qquad \phi(1)=0.
\]

The problem parameters are:

- Convection coefficient: \(a=1\)
- Diffusion coefficient: \(k=2\times10^{-4}\)
- Domain length: \(L=1\)

The corresponding Péclet number is

\[
\mathrm{Pe}=\frac{aL}{k}=5000.
\]

The high Péclet number indicates convection-dominated transport, making this problem a useful benchmark for evaluating numerical methods in the presence of sharp solution gradients.

## 2. Exact Solution

The analytical solution of the problem is

\[
\phi(x)=
\frac{1-\exp\!\left[\mathrm{Pe}(x-1)\right]}
{1-\exp(-\mathrm{Pe})}.
\]

This exact solution serves as a reference for evaluating the numerical approximations.

## 3. Files and Results

| File | Description |
|---|---|
| `P1.py` | Python script implementing the Legendre-based Modified Extreme Learning Machine (MELM) method. |
| `P1_PINN.csv` | PINN numerical data file required to run the Python script. |
| `P1_MELM_error.pdf` | Error plot for the MELM solution. |
| `P1_PINN_error.pdf` | Error plot for the PINN solution. |
| `P1_comparison.pdf` | Comparison of the MELM and PINN solutions. |
| `P1_zoom.pdf` | Zoomed-in comparison plot for examining solution details. |

## 4. How to Run

Ensure that `P1.py` and the required `P1_PIN
