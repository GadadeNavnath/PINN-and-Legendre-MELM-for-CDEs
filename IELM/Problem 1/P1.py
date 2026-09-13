#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legvander, legder, legval
import pandas as pd
import time
# ============================================================
# Problem parameters
# ============================================================

a = 1.0
k = 2.0e-4
L = 1.0

Pe = a * L / k

M = 375
N_int = 500

print("a  :", a)
print("k  :", k)
print("L  :", L)
print("Pe :", Pe)
print("Collocation Points:", N_int + 2)
print("Basis Functions:", M)

# ============================================================
# Exact solution
# ============================================================

def y_exact(x, Pe):

    return (
        1.0 - np.exp(Pe * (x - 1.0))
    ) / (
        1.0 - np.exp(-Pe)
    )

# ============================================================
# Precompute derivative coefficients
# ============================================================

d1_coeffs = []
d2_coeffs = []

for n in range(M):

    c = np.zeros(n + 1)
    c[-1] = 1.0

    d1_coeffs.append(legder(c, 1))
    d2_coeffs.append(legder(c, 2))

# ============================================================
# Basis matrices
# ============================================================

def Phi(x):

    xi = 2.0 * x / L - 1.0

    return legvander(xi, M - 1)


def Phi_x(x):

    xi = 2.0 * x / L - 1.0

    A = np.empty((len(x), M))

    for j in range(M):

        A[:, j] = (
            2.0 / L
        ) * legval(
            xi,
            d1_coeffs[j]
        )

    return A


def Phi_xx(x):

    xi = 2.0 * x / L - 1.0

    A = np.empty((len(x), M))

    for j in range(M):

        A[:, j] = (
            4.0 / L**2
        ) * legval(
            xi,
            d2_coeffs[j]
        )

    return A


start = time.time()

# ============================================================
# Collocation points : Chebyshev-Lobatto
# ============================================================

N_total = N_int + 2

j = np.arange(N_total)

x_cheb = (
    L / 2.0
    * (
        1.0
        - np.cos(
            np.pi * j / (N_total - 1)
        )
    )
)

x_int = x_cheb[1:-1]

# ============================================================
# Basis evaluation
# ============================================================

Phi_x_int = Phi_x(x_int)
Phi_xx_int = Phi_xx(x_int)

Phi_bc0 = Phi(np.array([0.0]))
Phi_bc1 = Phi(np.array([L]))

# ============================================================
# PDE matrix
# ============================================================

A_pde = (
    a * Phi_x_int
    - k * Phi_xx_int
)

b_pde = np.zeros(N_int)

# ============================================================
# Boundary conditions
# ============================================================

b_bc0 = np.array([1.0])
b_bc1 = np.array([0.0])

# ============================================================
# Assemble system
# ============================================================

A = np.empty(
    (N_int + 2, M)
)

A[:N_int] = A_pde
A[N_int] = Phi_bc0
A[N_int + 1] = Phi_bc1

b = np.concatenate(
    (
        b_pde,
        b_bc0,
        b_bc1
    )
)

# ============================================================
# Solve
# ============================================================

beta, residuals, rank, s = np.linalg.lstsq(
    A,
    b,
    rcond=None
)

# ============================================================
# Test points
# ============================================================

pinn = pd.read_csv(
    "P1_PINN.csv"
)

x_test = pinn["X"].to_numpy()

y_pinn = pinn["PINN"].to_numpy()

# ============================================================
# Exact solution
# ============================================================

y_ex = y_exact(
    x_test,
    Pe
)

# ============================================================
# IELM prediction
# ============================================================

Phi_test = Phi(x_test)

y_pred = Phi_test @ beta

end = time.time()

# ============================================================
# Errors
# ============================================================

err_ielm = np.abs(
    y_pred - y_ex
)

err_pinn = np.abs(
    y_pinn - y_ex
)

# ============================================================
# IELM errors
# ============================================================

Linf_ielm = np.max(
    err_ielm
)

RMS_ielm = np.sqrt(
    np.mean(
        (y_pred - y_ex) ** 2
    )
)

RelL2_ielm = (
    np.linalg.norm(
        y_pred - y_ex
    )
    /
    np.linalg.norm(y_ex)
)

# ============================================================
# PINN errors
# ============================================================

Linf_pinn = np.max(
    err_pinn
)

RMS_pinn = np.sqrt(
    np.mean(
        (y_pinn - y_ex) ** 2
    )
)

RelL2_pinn = (
    np.linalg.norm(
        y_pinn - y_ex
    )
    /
    np.linalg.norm(y_ex)
)

# ============================================================
# Results
# ============================================================

print("\n================ IELM ================")

print(
    f"Time taken      : {end-start:.2f} sec"
)

print(
    f"Linf error      : {Linf_ielm:.2e}"
)

print(
    f"RMS error       : {RMS_ielm:.2e}"
)

print(
    f"Relative L2 err : {RelL2_ielm:.2e}"
)

print("\n================ PINN ================")

print(
    f"Linf error      : {Linf_pinn:.2e}"
)

print(
    f"RMS error       : {RMS_pinn:.2e}"
)

print(
    f"Relative L2 err : {RelL2_pinn:.2e}"
)

# ============================================================
# Boundary values
# ============================================================

y0 = (
    Phi_bc0 @ beta
)[0]

y1 = (
    Phi_bc1 @ beta
)[0]

print("\nBoundary values")

print(
    f"phi(0) = {y0:.2e}"
)

print(
    f"phi(1) = {y1:.2e}"
)

# ============================================================
# Figure 1 : Comparison
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    x_test,
    y_ex,
    'k',
    lw=1.5,
    label='Exact'
)

plt.plot(
    x_test,
    y_pinn,
    '--',
    color='red',
    lw=1.5,
    label='PINN'
)

plt.plot(
    x_test,
    y_pred,
    '-.',
    color='blue',
    lw=1.5,
    label='IELM'
)

plt.xlabel("x")
plt.ylabel(r"$\phi(x)$")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "P1_comparison.pdf",
    dpi=300
)

plt.show()

# ============================================================
# Figure 2 : Zoom
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    x_test,
    y_ex,
    'k',
    lw=1.5,
    label='Exact'
)

plt.plot(
    x_test,
    y_pinn,
    '--',
    color='red',
    lw=1.5,
    label='PINN'
)

plt.plot(
    x_test,
    y_pred,
    '-.',
    color='blue',
    lw=1.5,
    label='IELM'
)

plt.xlim(
    0.998,
    1.0
)

plt.xlabel("x")
plt.ylabel(r"$\phi(x)$")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "P1_zoom.pdf",
    dpi=300
)

plt.show()

# ============================================================
# Figure 3 : PINN Absolute Error
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.semilogy(
    x_test,
    err_pinn,
    color='blue',
    lw=1.5
)

plt.xlabel("x")
plt.ylabel("Absolute error")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "P1_PINN_error.pdf",
    dpi=300
)

plt.show()

# ============================================================
# Figure 4 : IELM Absolute Error
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.semilogy(
    x_test,
    err_ielm,
    color='blue',
    lw=1.5
)

plt.xlabel("x")
plt.ylabel("Absolute error")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "P1_IELM_error.pdf",
    dpi=300
)

plt.show()


# In[ ]:




