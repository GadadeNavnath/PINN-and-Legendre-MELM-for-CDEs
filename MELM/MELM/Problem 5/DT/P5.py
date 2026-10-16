#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.polynomial.legendre import legval, legder
import time
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
import pandas as pd

# ==========================================================
# Domain
# ==========================================================

Lx = 2.0
Ly = 2.0
T  = 1.0
k = 0.01
U = 0.8

Pe = U*Lx/k

print(f"Pe = {Pe:.2e}")

print("Solving:")
print(f"u_t + 0.8*u_x + 0.8*u_y - {k}*(u_xx+u_yy) = 0")
print("")


# ==========================================================
# Basis size
# ==========================================================

Nx_basis = 50
Ny_basis = 50

Nbasis = Nx_basis * Ny_basis

print("Unknowns =", Nbasis)


# ==========================================================
# Legendre coefficients
# ==========================================================

Nmax = max(
    Nx_basis,
    Ny_basis
)

basis_coeffs = []
d1_coeffs = []
d2_coeffs = []

for n in range(Nmax):

    c = [0]*n + [1]

    basis_coeffs.append(c)

    d1_coeffs.append(
        legder(c)
    )

    d2_coeffs.append(
        legder(
            legder(c)
        )
    )


# ==========================================================
# Optimized 2D Basis matrix
# ==========================================================

def basis_matrix_2d(x, y):

    Np = len(x)

    xh = 2.0*x/Lx - 1.0
    yh = 2.0*y/Ly - 1.0

    # --------------------------------------------
    # Precompute all Legendre basis values
    # --------------------------------------------

    Px = np.empty(
        (Np, Nx_basis)
    )

    for i in range(Nx_basis):

        Px[:, i] = legval(
            xh,
            basis_coeffs[i]
        )

    Py = np.empty(
        (Np, Ny_basis)
    )

    for j in range(Ny_basis):

        Py[:, j] = legval(
            yh,
            basis_coeffs[j]
        )

    # --------------------------------------------
    # Assemble tensor-product basis
    # --------------------------------------------

    Phi = np.empty(
        (Np, Nbasis)
    )

    col = 0

    for i in range(Nx_basis):

        Pxi = Px[:, i]

        for j in range(Ny_basis):

            Phi[:, col] = Pxi * Py[:, j]

            col += 1

    return Phi


# ==========================================================
# Optimized Derivative matrices
# ==========================================================

def derivative_matrices_2d(x, y):

    Np = len(x)

    xh = 2.0*x/Lx - 1.0
    yh = 2.0*y/Ly - 1.0

    sx = 2.0/Lx
    sy = 2.0/Ly

    # --------------------------------------------------
    # Precompute all x-direction basis functions
    # --------------------------------------------------

    Px = np.empty(
        (Np, Nx_basis)
    )

    Pxd = np.empty(
        (Np, Nx_basis)
    )

    Pxdd = np.empty(
        (Np, Nx_basis)
    )

    for i in range(Nx_basis):

        Px[:, i] = legval(
            xh,
            basis_coeffs[i]
        )

        Pxd[:, i] = (
            sx *
            legval(
                xh,
                d1_coeffs[i]
            )
        )

        Pxdd[:, i] = (
            sx**2 *
            legval(
                xh,
                d2_coeffs[i]
            )
        )

    # --------------------------------------------------
    # Precompute all y-direction basis functions
    # --------------------------------------------------

    Py = np.empty(
        (Np, Ny_basis)
    )

    Pyd = np.empty(
        (Np, Ny_basis)
    )

    Pydd = np.empty(
        (Np, Ny_basis)
    )

    for j in range(Ny_basis):

        Py[:, j] = legval(
            yh,
            basis_coeffs[j]
        )

        Pyd[:, j] = (
            sy *
            legval(
                yh,
                d1_coeffs[j]
            )
        )

        Pydd[:, j] = (
            sy**2 *
            legval(
                yh,
                d2_coeffs[j]
            )
        )

    # --------------------------------------------------
    # Allocate matrices
    # --------------------------------------------------

    Dx = np.empty(
        (Np, Nbasis)
    )

    Dy = np.empty(
        (Np, Nbasis)
    )

    Dxx = np.empty(
        (Np, Nbasis)
    )

    Dyy = np.empty(
        (Np, Nbasis)
    )

    # --------------------------------------------------
    # Assemble tensor-product derivatives
    # --------------------------------------------------

    col = 0

    for i in range(Nx_basis):

        Px_i = Px[:, i]
        Pxd_i = Pxd[:, i]
        Pxdd_i = Pxdd[:, i]

        for j in range(Ny_basis):

            Py_j = Py[:, j]
            Pyd_j = Pyd[:, j]
            Pydd_j = Pydd[:, j]

            Dx[:, col] = (
                Pxd_i * Py_j
            )

            Dy[:, col] = (
                Px_i * Pyd_j
            )

            Dxx[:, col] = (
                Pxdd_i * Py_j
            )

            Dyy[:, col] = (
                Px_i * Pydd_j
            )

            col += 1

    return Dx, Dy, Dxx, Dyy


# ==========================================================
# Exact solution
# ==========================================================

ax_vel = 0.8
ay_vel = 0.8


def u_exact(x, y, t):

    sigma = 4.0*t + 1.0

    return (
        1.0/sigma
    ) * np.exp(
        -(1.0/k) * (
            (x - ax_vel*t - 0.5)**2
            +
            (y - ay_vel*t - 0.5)**2
        ) / sigma
    )


# ==========================================================
# Tensor-product Chebyshev-Gauss points
# ==========================================================

Nx_col = 50
Ny_col = 50

ix = np.arange(Nx_col)
iy = np.arange(Ny_col)

x_cheb = 0.5*Lx*(
    np.cos(
        (2*ix+1)*np.pi/(2*Nx_col)
    )
    + 1.0
)

y_cheb = 0.5*Ly*(
    np.cos(
        (2*iy+1)*np.pi/(2*Ny_col)
    )
    + 1.0
)


# ==========================================================
# Interior collocation points
# Chebyshev-Gauss points do not contain the endpoints.
# Therefore, ALL 50 x 50 points are interior points.
# ==========================================================

Xi, Yi = np.meshgrid(
    x_cheb,
    y_cheb,
    indexing='ij'
)

x_int = Xi.ravel()
y_int = Yi.ravel()

print(
    "Interior points =",
    len(x_int)
)


# ==========================================================
# Test grid
# ==========================================================

Nx_test = 101
Ny_test = 101

x_test = np.linspace(
    0.0,
    Lx,
    Nx_test
)

y_test = np.linspace(
    0.0,
    Ly,
    Ny_test
)

X_test, Y_test = np.meshgrid(
    x_test,
    y_test,
    indexing='ij'
)

x_test_flat = X_test.ravel()
y_test_flat = Y_test.ravel()

P_test = basis_matrix_2d(
    x_test_flat,
    y_test_flat
)


# ==========================================================
# Storage for selected solution profiles
# ==========================================================

solution_profiles = {}
exact_profiles = {}


# ==========================================================
# Start timer
# ==========================================================

start_time = time.time()


# ==========================================================
# Basis matrices
# ==========================================================

P = basis_matrix_2d(
    x_int,
    y_int
)

Dx, Dy, Dxx, Dyy = derivative_matrices_2d(
    x_int,
    y_int
)


# ==========================================================
# Left boundary (x=0)
# ==========================================================

yb = y_cheb

x_left = np.zeros_like(yb)

P_left = basis_matrix_2d(
    x_left,
    yb
)


# ==========================================================
# Right boundary (x=Lx)
# ==========================================================

x_right = Lx*np.ones_like(yb)

P_right = basis_matrix_2d(
    x_right,
    yb
)


# ==========================================================
# Bottom boundary (y=0)
# ==========================================================

xb = x_cheb

y_bottom = np.zeros_like(xb)

P_bottom = basis_matrix_2d(
    xb,
    y_bottom
)


# ==========================================================
# Top boundary (y=Ly)
# ==========================================================

y_top = Ly*np.ones_like(xb)

P_top = basis_matrix_2d(
    xb,
    y_top
)


# ==========================================================
# Prescribed initial condition
# ==========================================================

u0 = u_exact(
    x_int,
    y_int,
    0.0
)


# ==========================================================
# Spatial differential operator
# ==========================================================

L = (
    ax_vel*Dx
    +
    ay_vel*Dy
    -
    k*(Dxx + Dyy)
)


# ==========================================================
# Time step
# ==========================================================

dt = 0.001

Nt = int(T/dt)


# ==========================================================
# Error storage for all time steps
# ==========================================================

time_history = np.zeros(Nt)

relative_L2_history = np.zeros(Nt)

max_absolute_history = np.zeros(Nt)


# ==========================================================
# ==========================================================
# BDF1 / Backward Euler STARTUP
#
# (u^(1) - u^(0))/dt + L u^(1) = 0
#
# Therefore:
#
# (P + dt L) beta^(1) = u^(0)
#
# ==========================================================
# ==========================================================

A_int = (
    P
    +
    dt*L
)

t = dt


# ----------------------------------------------------------
# Boundary conditions at t = dt
# ----------------------------------------------------------

u_left = u_exact(
    x_left,
    yb,
    t
)

u_right = u_exact(
    x_right,
    yb,
    t
)

u_bottom = u_exact(
    xb,
    y_bottom,
    t
)

u_top = u_exact(
    xb,
    y_top,
    t
)


# ----------------------------------------------------------
# BDF1 system
#
# u^(0) is used directly.
# ----------------------------------------------------------

A = np.vstack([

    P_left,

    A_int,

    P_right,

    P_bottom,

    P_top

])


rhs = np.concatenate([

    u_left,

    u0,

    u_right,

    u_bottom,

    u_top

])


beta = np.linalg.lstsq(
    A,
    rhs,
    rcond=None
)[0]


# ----------------------------------------------------------
# Numerical solution u^(1)
# ----------------------------------------------------------

u = P @ beta


# ==========================================================
# Evaluate u^(1) on test grid
# ==========================================================

u_test = (
    P_test @ beta
).reshape(
    Nx_test,
    Ny_test
)


# ==========================================================
# Exact solution at t = dt
# ==========================================================

u_exact_test = u_exact(
    X_test,
    Y_test,
    dt
)


# ==========================================================
# Overall Relative L2 Error
#
# Start accumulation from t = dt.
# t = 0 is excluded because u^(0) is prescribed.
# ==========================================================

err = (
    u_test
    -
    u_exact_test
)

sum_err_sq = np.sum(
    err**2
)

sum_exact_sq = np.sum(
    u_exact_test**2
)


# ==========================================================
# Store BDF1 error values
# ==========================================================

time_history[0] = t

relative_L2_history[0] = (
    np.linalg.norm(
        err.ravel()
    )
    /
    np.linalg.norm(
        u_exact_test.ravel()
    )
)

max_absolute_history[0] = np.max(
    np.abs(err)
)


# ==========================================================
# Initialize BDF2 history
#
# u_prev = u^(0)
# u_curr = u^(1)
# ==========================================================

u_prev = u0.copy()

u_curr = u.copy()


# ==========================================================
# BDF2 matrix
#
# (3/2 P + dt L) beta^(n+1)
#
# =
#
# 2u^(n) - (1/2)u^(n-1)
# ==========================================================

A_int = (
    1.5*P
    +
    dt*L
)


A = np.vstack([

    P_left,

    A_int,

    P_right,

    P_bottom,

    P_top

])


# ==========================================================
# QR factorization
# Performed only once
# ==========================================================

Q, R = np.linalg.qr(
    A,
    mode="reduced"
)

QT = Q.T


# ==========================================================
# Save solution at selected time steps
# ==========================================================

save_steps = {

    200: 0.20,

    400: 0.40,

    600: 0.60,

    800: 0.80,

    1000: 1.00

}


# ==========================================================
# BDF2 time integration
# ==========================================================

for n in range(1,Nt):
    t = (
        n + 1
    ) * dt


    # ------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------

    u_left = u_exact(
        x_left,
        yb,
        t
    )

    u_right = u_exact(
        x_right,
        yb,
        t
    )

    u_bottom = u_exact(
        xb,
        y_bottom,
        t
    )

    u_top = u_exact(
        xb,
        y_top,
        t
    )


    # ------------------------------------------------------
    # BDF2 interior RHS
    #
    # 2u^(n) - 0.5u^(n-1)
    # ------------------------------------------------------

    rhs_int = (
        2.0 * u_curr
        -
        0.5 * u_prev
    )


    # ------------------------------------------------------
    # Complete RHS
    # ------------------------------------------------------

    rhs = np.concatenate([

        u_left,

        rhs_int,

        u_right,

        u_bottom,

        u_top

    ])


    # ------------------------------------------------------
    # Solve BDF2 system
    # ------------------------------------------------------

    rhs_q = QT @ rhs

    beta = solve_triangular(
        R,
        rhs_q,
        lower=False,
        check_finite=False
    )


    # ------------------------------------------------------
    # New solution u^(n+1)
    # ------------------------------------------------------

    u_new = P @ beta


    # ------------------------------------------------------
    # Test-grid solution
    # ------------------------------------------------------

    u_test = (
        P_test @ beta
    ).reshape(
        Nx_test,
        Ny_test
    )


    # ------------------------------------------------------
    # Exact solution
    # ------------------------------------------------------

    u_exact_test = u_exact(
        X_test,
        Y_test,
        t
    )


    # ======================================================
    # Accumulate Relative L2 Error
    #
    # All computed time levels are included:
    #
    # t = 0.001, 0.002, ..., 1.000
    #
    # t = 0 is excluded.
    # ======================================================

    err = (
        u_test
        -
        u_exact_test
    )

    sum_err_sq += np.sum(
        err**2
    )

    sum_exact_sq += np.sum(
        u_exact_test**2
    )


    # ------------------------------------------------------
    # Store error values for current time step
    # ------------------------------------------------------
    step = n + 1

    time_history[step - 1] = t

    relative_L2_history[step - 1] = (
        np.linalg.norm(
            err.ravel()
        )
        /
        np.linalg.norm(
            u_exact_test.ravel()
        )
    )

    max_absolute_history[step - 1] = np.max(
        np.abs(err)
    )


    # ------------------------------------------------------
    # Save selected time levels
    # ------------------------------------------------------

    if step in save_steps:

        key = save_steps[step]

        solution_profiles[key] = (
            u_test.copy()
        )

        exact_profiles[key] = (
            u_exact_test.copy()
        )


    # ------------------------------------------------------
    # Update BDF2 history
    #
    # u_prev <- u^(n)
    # u_curr <- u^(n+1)
    # ------------------------------------------------------

    u_prev, u_curr = (
        u_curr,
        u_new
    )


# ==========================================================
# End timer
# ==========================================================

end_time = time.time()

print(
    f"\nTotal CPU time = "
    f"{end_time - start_time:.2f} seconds"
)


# ==========================================================
# Overall Relative L2 Error
#
# ALL computed time steps:
#
# t = 0.001, 0.002, ..., 1.000
#
# t = 0 is NOT included.
# ==========================================================

rel_L2 = (
    np.sqrt(sum_err_sq)
    /
    np.sqrt(sum_exact_sq)
)


print("=" * 60)

print(
    f"Overall Relative L2 Error "
    f"(all time steps) = {rel_L2:.2e}"
)

print("=" * 60)


# ==========================================================
# Save Relative L2 Error and Maximum Absolute Error
# for all time steps to CSV
# ==========================================================

error_df = pd.DataFrame({

    "Time": time_history,

    "Relative_L2_Error": relative_L2_history,

    "Max_Absolute_Error": max_absolute_history

})

error_df.to_csv(
    "P5_DT_MELM_errors.csv",
    index=False
)

print(
    "Saved P5_DT_MELM_errors.csv"
)


# ==========================================================
# Save numerical solution to CSV
#
# Only selected output times are saved.
# ==========================================================

rows = []


for tp in sorted(
    solution_profiles.keys()
):

    U_num = solution_profiles[tp]

    rows.extend(
        zip(
            X_test.ravel(),
            Y_test.ravel(),
            np.full(
                X_test.size,
                tp
            ),
            U_num.ravel()
        )
    )


df = pd.DataFrame(
    rows,
    columns=[
        "X",
        "Y",
        "T",
        "U"
    ]
)


df.to_csv(
    "P5_DT_MELM.csv",
    index=False
)


print(
    "Saved P5_DT_MELM.csv"
)


# ==========================================================
# Times to plot
# ==========================================================

plot_times = [

    0.20,

    0.40,

    0.60,

    0.80,

    1.00

]


# ==========================================================
# Compute errors at selected reporting times
# ==========================================================

for tp in plot_times:

    tp = round(
        tp,
        2
    )


    # -----------------------------------------
    # Exact solution
    # -----------------------------------------

    Ue = exact_profiles[tp]

    Un = solution_profiles[tp]


    # -----------------------------------------
    # Absolute error
    # -----------------------------------------

    Err = np.abs(
        Ue - Un
    )


    # -----------------------------------------
    # RMS error
    # -----------------------------------------

    RMS = np.sqrt(
        np.mean(
            Err**2
        )
    )


    # -----------------------------------------
    # Maximum error
    # -----------------------------------------

    Linf = np.max(
        Err
    )


    print(
        f"t = {tp:.2f}   "
        f"Linf = {Linf:.2e}   "
        f"RMS = {RMS:.2e}"
    )


# ==========================================================
# Final Overall Relative L2 Error
# ==========================================================

print("=" * 60)

print(
    f"Overall Relative L2 Error "
    f"(all time steps) = {rel_L2:.2e}"
)

print("=" * 60)


# In[ ]:




