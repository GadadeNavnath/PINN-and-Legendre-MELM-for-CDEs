#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.polynomial.legendre import legval, legder
import pandas as pd
import scipy.linalg
# ==========================================================
# Parameters
# ==========================================================

Pe = 1000.0

Nx = 109
Ny = 109

Nbasis = (Nx + 1) * (Ny + 1)

Nc = 110
Nb = 500

ax_vel = 1.0 / np.sqrt(5.0)
ay_vel = -2.0 / np.sqrt(5.0)

# ==========================================================
# Legendre coefficients
# ==========================================================

basis_coeffs = []
d1_coeffs = []
d2_coeffs = []

for n in range(max(Nx, Ny) + 1):

    c = [0] * n + [1]

    basis_coeffs.append(c)
    d1_coeffs.append(legder(c))
    d2_coeffs.append(legder(legder(c)))

# ==========================================================
# Precompute Legendre basis values
# ==========================================================

def precompute_legendre(x, N):

    x_hat = 2.0 * x

    P = np.empty((N + 1, len(x)))
    Px = np.empty((N + 1, len(x)))
    Pxx = np.empty((N + 1, len(x)))

    for n in range(N + 1):

        P[n] = legval(
            x_hat,
            basis_coeffs[n]
        )

        Px[n] = 2.0 * legval(
            x_hat,
            d1_coeffs[n]
        )

        Pxx[n] = 4.0 * legval(
            x_hat,
            d2_coeffs[n]
        )

    return P, Px, Pxx


# ==========================================================
# Build PDE operator matrix
# ==========================================================

def build_operator_matrix(x, y):

    Np = len(x)

    A = np.empty(
        (Np, Nbasis),
        order="F"
    )

    Px, Pxx_x, Pxxx = precompute_legendre(
        x,
        Nx
    )

    Py, Pyy_y, Pyyy = precompute_legendre(
        y,
        Ny
    )

    col = 0

    for i in range(Nx + 1):

        Pi = Px[i]
        Pix = Pxx_x[i]
        Pixx = Pxxx[i]

        for j in range(Ny + 1):

            Pj = Py[j]
            Pjy = Pyy_y[j]
            Pjyy = Pyyy[j]

            A[:, col] = (
                -(1.0 / Pe)
                * (
                    Pixx * Pj
                    + Pi * Pjyy
                )
                + ax_vel * Pix * Pj
                + ay_vel * Pi * Pjy
            )

            col += 1

    return A


# ==========================================================
# Build basis matrix
# ==========================================================

def build_basis_matrix(x, y):

    Np = len(x)

    Phi = np.empty(
        (Np, Nbasis),
        order="F"
    )

    Px, _, _ = precompute_legendre(
        x,
        Nx
    )

    Py, _, _ = precompute_legendre(
        y,
        Ny
    )

    col = 0

    for i in range(Nx + 1):

        Pi = Px[i]

        for j in range(Ny + 1):

            Phi[:, col] = Pi * Py[j]

            col += 1

    return Phi


# ==========================================================
# Interior points
# ==========================================================

start_time = time.time()

k = np.arange(Nc)

xc = 0.5 * np.cos(
    (2 * k + 1) * np.pi / (2 * Nc)
)

yc = 0.5 * np.cos(
    (2 * k + 1) * np.pi / (2 * Nc)
)

Xc, Yc = np.meshgrid(
    xc,
    yc
)

x_r = Xc.ravel()
y_r = Yc.ravel()

Nr = len(x_r)

print(f"Interior points : {Nr}")

A_pde = build_operator_matrix(
    x_r,
    y_r
)

b_pde = np.zeros(Nr)


# ==========================================================
# Boundary points
# ==========================================================

s = np.linspace(
    -0.5,
    0.5,
    Nb
)

x1 = np.full(Nb, -0.5)
y1 = s

x2 = np.full(Nb, 0.5)
y2 = s

x3 = s
y3 = np.full(Nb, -0.5)

x4 = s
y4 = np.full(Nb, 0.5)

xb = np.concatenate(
    (
        x1,
        x2,
        x3,
        x4
    )
)

yb = np.concatenate(
    (
        y1,
        y2,
        y3,
        y4
    )
)

print(
    f"Boundary points : {len(xb)}"
)

A_bc = build_basis_matrix(
    xb,
    yb
)


# ==========================================================
# Discontinuous Dirichlet boundary conditions
# ==========================================================

b_bc = np.zeros(
    len(xb)
)

b_bc[
    3 * Nb : 4 * Nb
] = 1.0

left_mask = (
    y1 >= 0.25
)

b_bc[:Nb][left_mask] = 1.0


# ==========================================================
# Least-squares system
# ==========================================================

w_bc = 1.0

A = np.vstack(
    (
        A_pde,
        w_bc * A_bc
    )
)

b = np.concatenate(
    (
        b_pde,
        w_bc * b_bc
    )
)

print(
    "System size :",
    A.shape
)


# ==========================================================
# Solve
# ==========================================================

print(
    "\nSolving least-squares system ..."
)

beta = scipy.linalg.lstsq(
    A,
    b,
    lapack_driver="gelsy"
)[0]


end_time = time.time()

print(
    f"Time taken : "
    f"{end_time - start_time:.2f} sec"
)


# ==========================================================
# PDE residual
# ==========================================================

residual = (
    A_pde @ beta
    - b_pde
)

Linf_res = np.max(
    np.abs(residual)
)

L2_res = np.sqrt(
    np.mean(residual ** 2)
)

print("\nPDE Residuals")

print(
    f"Linf error = {Linf_res:.2e}"
)

print(
    f"RMS  error = {L2_res:.2e}"
)


# ==========================================================
# Boundary check
# ==========================================================

ub = A_bc @ beta

bc_error = (
    ub - b_bc
)

Linf_bc = np.max(
    np.abs(bc_error)
)

L2_bc = np.sqrt(
    np.mean(bc_error ** 2)
)


print("\nBoundary Errors")

print(
    f"Linf error = {Linf_bc:.2e}"
)

print(
    f"RMS  error = {L2_bc:.2e}"
)

# ==========================================================
# Evaluation points
# ==========================================================

ref = pd.read_csv(
    "P3_PINN.csv",
    sep=";"
)

X = ref["X"].to_numpy()
Y = ref["Y"].to_numpy()

print(
    f"Evaluation points : {len(X)}"
)


# ==========================================================
# Evaluate MELM solution
# ==========================================================

Phi_eval = build_basis_matrix(
    X,
    Y
)

U = Phi_eval @ beta


# ==========================================================
# Save solution
# ==========================================================

data = np.column_stack(
    (
        X,
        Y,
        U
    )
)

np.savetxt(
    "P3_MELM.csv",
    data,
    delimiter=",",
    header="X,Y,U",
    comments=""
)

print(
    "Solution saved to P3_MELM.csv"
)

