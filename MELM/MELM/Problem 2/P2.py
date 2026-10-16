#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

ax_vel = 1.0 / np.sqrt(2)
ay_vel = 1.0 / np.sqrt(2)

Nx = 109
Ny = 109

Nbasis = (Nx + 1) * (Ny + 1)

Nc = 110
Nb = 500

print("Pe:", Pe)
print("a_x:", ax_vel)
print("a_y:", ay_vel)
print("k:", 1.0 / Pe)
print("Basis Functions:", Nbasis)

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
# PDE operator matrix
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
                    +
                    Pi * Pjyy
                )

                + ax_vel
                * Pix
                * Pj

                + ay_vel
                * Pi
                * Pjy
            )

            col += 1

    return A


# ==========================================================
# Basis matrix
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

start = time.time()

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

A_pde = build_operator_matrix(
    x_r,
    y_r
)

b_pde = 5.0 * np.ones(Nr)


# ==========================================================
# Boundary points
# ==========================================================

s = np.linspace(
    -0.5,
    0.5,
    Nb
)

x1 = -0.5 * np.ones_like(s)
y1 = s

x2 = 0.5 * np.ones_like(s)
y2 = s

x3 = s
y3 = -0.5 * np.ones_like(s)

x4 = s
y4 = 0.5 * np.ones_like(s)

xb = np.concatenate([
    x1,
    x2,
    x3,
    x4
])

yb = np.concatenate([
    y1,
    y2,
    y3,
    y4
])

A_bc = build_basis_matrix(
    xb,
    yb
)

b_bc = np.zeros(
    len(xb)
)

print("Interior points :", len(x_r))
print("Boundary points :", len(xb))


# ==========================================================
# Least-squares system
# ==========================================================

w_bc = 1.0

A = np.vstack([
    A_pde,
    w_bc * A_bc
])

b = np.concatenate([
    b_pde,
    w_bc * b_bc
])

print("System size :", A.shape)


# ==========================================================
# Solve
# ==========================================================

beta = scipy.linalg.lstsq(
    A,
    b,
    lapack_driver="gelsy"
)[0]

# ==========================================================
# PDE residual check
# ==========================================================

residual = (
    A_pde @ beta
    - b_pde
)

L2_res = np.sqrt(
    np.mean(residual ** 2)
)

Linf_res = np.max(
    np.abs(residual)
)

print("\nPDE residuals")

print(
    f"Linf error = {Linf_res:.2e}"
)

print(
    f"RMS error   = {L2_res:.2e}"
)


# ==========================================================
# Boundary check
# ==========================================================

ub = A_bc @ beta

end = time.time()

print("\nBoundary errors")

print(
    "Time taken:",
    np.round(end - start, 2),
    "Sec"
)

print(
    f"\nLinf error = "
    f"{np.max(np.abs(ub)):.2e}"
)

print(
    f"RMS error   = "
    f"{np.sqrt(np.mean(ub ** 2)):.2e}"
)

print(
    f"Condition number of A = "
    f"{np.linalg.cond(A):.2e}"
)

# ==========================================================
# Evaluation points
# ==========================================================

ref = pd.read_csv(
    "P2_PINN.csv",
    sep=';'
)

X = ref["X"].to_numpy()
Y = ref["Y"].to_numpy()

print(
    f"Evaluation points : {len(X)}"
)


# ==========================================================
# Evaluate MELM solution
# ==========================================================

Phi_plot = build_basis_matrix(
    X,
    Y
)

U = Phi_plot @ beta


# ==========================================================
# Save physical solution
# ==========================================================

data = np.column_stack((
    X,
    Y,
    U
))

np.savetxt(
    "P2_MELM.csv",
    data,
    delimiter=",",
    header="X,Y,U",
    comments=""
)

print(
    "Solution saved to P2_MELM.csv"
)


# In[ ]:




