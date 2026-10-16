#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.polynomial.legendre import legval, legder
import time
import scipy.linalg

# =====================================================
# Start timer
# =====================================================

start = time.time()

# =====================================================
# Exact solution
# =====================================================

a = 1.0
k = 0.1

def u_exact(x, t):

    return (
        1.0 / np.sqrt(4.0 * t + 1.0)
    ) * np.exp(
        -((x - 1.0 - a * t) ** 2)
        /
        (k * (4.0 * t + 1.0))
    )

# =====================================================
# Domain
# =====================================================

Lx = 9.0
T = 5.0

# =====================================================
# Basis parameters
# =====================================================

Nx_basis = 95
Nt_basis = 85

N_unknown = Nx_basis * Nt_basis

print(f"Unknowns : {N_unknown}")

# =====================================================
# Chebyshev interior points
# =====================================================

Nx_col = 100
Nt_col = 90

kx = np.arange(Nx_col)
kt = np.arange(Nt_col)

x_cheb = 0.5 * Lx * (
    np.cos((2 * kx + 1) * np.pi / (2 * Nx_col))
    + 1.0
)

t_cheb = 0.5 * T * (
    np.cos((2 * kt + 1) * np.pi / (2 * Nt_col))
    + 1.0
)

# =====================================================
# PDE time points
# Interior Chebyshev-Gauss points + final time t = T
# =====================================================

t_pde = np.concatenate([
    t_cheb,
    np.array([T])
])

# =====================================================
# PDE collocation points
# =====================================================

Xc, Tc = np.meshgrid(
    x_cheb,
    t_pde
)

x_int = Xc.ravel()
t_int = Tc.ravel()

print(f"Interior PDE points : {len(x_int)}")

# =====================================================
# Legendre coefficients
# =====================================================

basis_coeffs = []
d1_coeffs = []
d2_coeffs = []

for n in range(max(Nx_basis, Nt_basis)):

    c = [0] * n + [1]

    basis_coeffs.append(c)

    d1_coeffs.append(
        legder(c)
    )

    d2_coeffs.append(
        legder(
            legder(c)
        )
    )

# =====================================================
# Basis matrix
# =====================================================

def basis_matrix(x, t):

    xh = 2.0 * x / Lx - 1.0
    th = 2.0 * t / T - 1.0

    cols = []

    for i in range(Nx_basis):

        Px = legval(
            xh,
            basis_coeffs[i]
        )

        for j in range(Nt_basis):

            Pt = legval(
                th,
                basis_coeffs[j]
            )

            cols.append(
                Px * Pt
            )

    return np.vstack(cols).T

# =====================================================
# Derivative matrices
# =====================================================

def derivative_matrices(x, t):

    xh = 2.0 * x / Lx - 1.0
    th = 2.0 * t / T - 1.0

    sx = 2.0 / Lx
    st = 2.0 / T

    dx_cols = []
    dxx_cols = []
    dt_cols = []

    for i in range(Nx_basis):

        Px = legval(
            xh,
            basis_coeffs[i]
        )

        dPx = sx * legval(
            xh,
            d1_coeffs[i]
        )

        d2Px = sx ** 2 * legval(
            xh,
            d2_coeffs[i]
        )

        for j in range(Nt_basis):

            Pt = legval(
                th,
                basis_coeffs[j]
            )

            dPt = st * legval(
                th,
                d1_coeffs[j]
            )

            dx_cols.append(
                dPx * Pt
            )

            dxx_cols.append(
                d2Px * Pt
            )

            dt_cols.append(
                Px * dPt
            )

    return (
        np.vstack(dx_cols).T,
        np.vstack(dxx_cols).T,
        np.vstack(dt_cols).T
    )

# =====================================================
# PDE matrix
# =====================================================

Px_int, Pxx_int, Pt_int = derivative_matrices(
    x_int,
    t_int
)

A_pde = (
    Pt_int
    +
    a * Px_int
    -
    k * Pxx_int
)

b_pde = np.zeros(
    len(x_int)
)

print(f"PDE points : {len(b_pde)}")

# =====================================================
# Initial condition
# t = 0
# =====================================================

x_ic = x_cheb

t_ic = np.zeros_like(
    x_ic
)

P_ic = basis_matrix(
    x_ic,
    t_ic
)

u_ic = u_exact(
    x_ic,
    t_ic
)

# =====================================================
# Boundary time points
# t = 0 + interior Chebyshev-Gauss + t = T
# =====================================================

t_bc = np.concatenate([
    np.array([0.0]),
    t_cheb,
    np.array([T])
])

# =====================================================
# Left boundary x = 0
# =====================================================

x_left = np.zeros_like(
    t_bc
)

P_left = basis_matrix(
    x_left,
    t_bc
)

u_left = u_exact(
    x_left,
    t_bc
)

# =====================================================
# Right boundary x = Lx
# =====================================================

x_right = Lx * np.ones_like(
    t_bc
)

P_right = basis_matrix(
    x_right,
    t_bc
)

u_right = u_exact(
    x_right,
    t_bc
)

# =====================================================
# Point counts
# =====================================================

print(f"Initial points : {len(x_ic)}")
print(f"Boundary time points : {len(t_bc)}")
print(f"Left boundary points : {len(x_left)}")
print(f"Right boundary points : {len(x_right)}")

# =====================================================
# Assemble least-squares system
# =====================================================

A = np.vstack([
    A_pde,
    P_ic,
    P_left,
    P_right
])

b = np.concatenate([
    b_pde,
    u_ic,
    u_left,
    u_right
])

print("\n=====================================================")
print("Linear system")
print("=====================================================")

print(f"System size : {A.shape}")

# =====================================================
# Solve least-squares system using GELSY
# =====================================================

print("\nSolving least-squares system ...")

solve_start = time.time()

beta = scipy.linalg.lstsq(
    A,
    b,
    lapack_driver="gelsy"
)[0]

solve_end = time.time()

print(
    f"Solve time : {solve_end - solve_start:.2f} sec"
)

# =====================================================
# Total elapsed time
# =====================================================

end = time.time()

print(
    f"Total time : {end - start:.2f} sec"
)

# =====================================================
# PDE residual check
# =====================================================

residual = A_pde @ beta - b_pde

Linf_res = np.max(
    np.abs(residual)
)

L2_res = np.sqrt(
    np.mean(residual**2)
)

print("\n=====================================================")
print("PDE Residuals")
print("=====================================================")

print(
    f"Linf error = {Linf_res:.2e}"
)

print(
    f"RMS  error = {L2_res:.2e}"
)

# =====================================================
# Initial condition check
# =====================================================

u_ic_pred = P_ic @ beta

ic_error = u_ic_pred - u_ic

Linf_ic = np.max(
    np.abs(ic_error)
)

L2_ic = np.sqrt(
    np.mean(ic_error**2)
)

print("\n=====================================================")
print("Initial Condition Errors")
print("=====================================================")

print(
    f"Linf error = {Linf_ic:.2e}"
)

print(
    f"RMS  error = {L2_ic:.2e}"
)

# =====================================================
# Left boundary check
# =====================================================

u_left_pred = P_left @ beta

left_error = u_left_pred - u_left

Linf_left = np.max(
    np.abs(left_error)
)

L2_left = np.sqrt(
    np.mean(left_error**2)
)

print("\n=====================================================")
print("Left Boundary Errors")
print("=====================================================")

print(
    f"Linf error = {Linf_left:.2e}"
)

print(
    f"RMS  error = {L2_left:.2e}"
)

# =====================================================
# Right boundary check
# =====================================================

u_right_pred = P_right @ beta

right_error = u_right_pred - u_right

Linf_right = np.max(
    np.abs(right_error)
)

L2_right = np.sqrt(
    np.mean(right_error**2)
)

print("\n=====================================================")
print("Right Boundary Errors")
print("=====================================================")

print(
    f"Linf error = {Linf_right:.2e}"
)

print(
    f"RMS  error = {L2_right:.2e}"
)

# =====================================================
# Test grid
# =====================================================

Nx_test = 201
Nt_test = 101

x_test = np.linspace(
    0.0,
    Lx,
    Nx_test
)

t_test = np.linspace(
    0.0,
    T,
    Nt_test
)

X, TT = np.meshgrid(
    x_test,
    t_test
)

print("\n=====================================================")
print("Test evaluation")
print("=====================================================")

print(
    f"Evaluation points : {X.size}"
)

# =====================================================
# Evaluate Legendre approximation
# =====================================================

P_test = basis_matrix(
    X.ravel(),
    TT.ravel()
)

u_pred = (
    P_test @ beta
).reshape(
    Nt_test,
    Nx_test
)

# =====================================================
# Exact solution
# =====================================================

u_ex = u_exact(
    X,
    TT
)

# =====================================================
# Numerical errors
# =====================================================

err = np.abs(
    u_pred - u_ex
)

RMS = np.sqrt(
    np.mean(err**2)
)

Linf = np.max(
    err
)

Rel_L2 = (
    np.linalg.norm(err)
    /
    np.linalg.norm(u_ex)
)

print("\n=====================================================")
print("Solution Errors")
print("=====================================================")

print(
    f"RMS  error = {RMS:.2e}"
)

print(
    f"Linf error = {Linf:.2e}"
)

print(
    f"Relative L2 error = {Rel_L2:.2e}"
)

# =====================================================
# Error at t = 0
# =====================================================

idx_t0 = 0

err_t0 = np.abs(
    u_pred[idx_t0, :]
    -
    u_ex[idx_t0, :]
)

RMS_t0 = np.sqrt(
    np.mean(err_t0**2)
)

Linf_t0 = np.max(
    err_t0
)

Rel_L2_t0 = (
    np.linalg.norm(err_t0)
    /
    np.linalg.norm(u_ex[idx_t0, :])
)

print("\n-----------------------------------------------------")
print("Errors at t = 0")
print("-----------------------------------------------------")

print(
    f"RMS  error = {RMS_t0:.2e}"
)

print(
    f"Linf error = {Linf_t0:.2e}"
)

print(
    f"Relative L2 error = {Rel_L2_t0:.2e}"
)

# =====================================================
# Error at t = T = 5
# =====================================================

idx_T = -1

err_T = np.abs(
    u_pred[idx_T, :]
    -
    u_ex[idx_T, :]
)

RMS_T = np.sqrt(
    np.mean(err_T**2)
)

Linf_T = np.max(
    err_T
)

Rel_L2_T = (
    np.linalg.norm(err_T)
    /
    np.linalg.norm(u_ex[idx_T, :])
)

print("\n-----------------------------------------------------")
print("Errors at t = 5")
print("-----------------------------------------------------")

print(
    f"RMS  error = {RMS_T:.2e}"
)

print(
    f"Linf error = {Linf_T:.2e}"
)

print(
    f"Relative L2 error = {Rel_L2_T:.2e}"
)

# =====================================================
# Save solution
# =====================================================

data = np.column_stack((
    X.ravel(),
    TT.ravel(),
    u_pred.ravel()
))

np.savetxt(
    "P4_CT_MELM.csv",
    data,
    delimiter=",",
    header="X,T,U",
    comments=""
)

print("\n=====================================================")
print("Solution saved")
print("=====================================================")

print(
    "Filename : P4_CT_MELM.csv"
)

print(
    f"Total stored points : {len(data)}"
)

print(
    f"Job finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}"
)


# In[ ]:




