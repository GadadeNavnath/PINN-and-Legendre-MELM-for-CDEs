#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.polynomial.legendre import legval, legder
import time
import scipy.linalg

# ==========================================================
# Problem parameters
# ==========================================================

Lx = 2.0
Ly = 2.0
T  = 1.0

k = 0.01
U = 0.8

Pe = U * Lx / k

print(f"Peclet number = {Pe:.2f}")

# ==========================================================
# Basis size
# ==========================================================

Nx_basis = 48
Ny_basis = 48
Nt_basis = 23

Nbasis = Nx_basis * Ny_basis * Nt_basis

print("=" * 60)
print("3D Legendre MELM")
print("=" * 60)
print(f"Basis functions : {Nbasis}")
print()

# ==========================================================
# Exact solution parameters
# ==========================================================

ax_vel = 0.8
ay_vel = 0.8

t0 = time.time()

# ==========================================================
# Exact solution
# ==========================================================

def u_exact(x, y, t):

    sigma = 4.0 * t + 1.0

    return (
        1.0 / sigma
    ) * np.exp(
        -(
            (x - ax_vel * t - 0.5) ** 2
            +
            (y - ay_vel * t - 0.5) ** 2
        ) / (k * sigma)
    )


# ==========================================================
# Precompute Legendre coefficients
# ==========================================================

Nmax = max(
    Nx_basis,
    Ny_basis,
    Nt_basis
)

basis_coeffs = []
d1_coeffs = []
d2_coeffs = []

for n in range(Nmax):

    c = np.zeros(n + 1)
    c[-1] = 1.0

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
# Precompute Legendre values
# ==========================================================

def precompute_legendre(
    x,
    N,
    scale
):

    x = np.asarray(x)

    P = np.empty((N, len(x)))
    Px = np.empty((N, len(x)))
    Pxx = np.empty((N, len(x)))

    for i in range(N):

        P[i] = legval(
            x,
            basis_coeffs[i]
        )

        Px[i] = (
            scale
            *
            legval(
                x,
                d1_coeffs[i]
            )
        )

        Pxx[i] = (
            scale ** 2
            *
            legval(
                x,
                d2_coeffs[i]
            )
        )

    return (
        P,
        Px,
        Pxx
    )


# ==========================================================
# Optimized basis matrix
# ==========================================================

def basis_matrix_3d(
    x,
    y,
    t
):

    Np = len(x)

    Phi = np.empty(
        (Np, Nbasis),
        order="F"
    )

    xh = 2.0 * x / Lx - 1.0
    yh = 2.0 * y / Ly - 1.0
    th = 2.0 * t / T  - 1.0

    Px, _, _ = precompute_legendre(
        xh,
        Nx_basis,
        2.0 / Lx
    )

    Py, _, _ = precompute_legendre(
        yh,
        Ny_basis,
        2.0 / Ly
    )

    Pt, _, _ = precompute_legendre(
        th,
        Nt_basis,
        2.0 / T
    )

    col = 0

    for i in range(Nx_basis):

        Pi = Px[i]

        for j in range(Ny_basis):

            Pj = Py[j]

            Pxy = Pi * Pj

            for k_ in range(Nt_basis):

                Phi[:, col] = (
                    Pxy
                    *
                    Pt[k_]
                )

                col += 1

    return Phi


# ==========================================================
# PDE operator matrix
# ==========================================================

def pde_operator_matrix_3d(
    x,
    y,
    t
):

    Np = len(x)

    A_pde = np.empty(
        (Np, Nbasis),
        dtype=np.float64,
        order="F"
    )

    xh = 2.0 * x / Lx - 1.0
    yh = 2.0 * y / Ly - 1.0
    th = 2.0 * t / T - 1.0

    Px, Pxd, Pxdd = precompute_legendre(
        xh,
        Nx_basis,
        2.0 / Lx
    )

    Py, Pyd, Pydd = precompute_legendre(
        yh,
        Ny_basis,
        2.0 / Ly
    )

    Pt, Ptd, _ = precompute_legendre(
        th,
        Nt_basis,
        2.0 / T
    )

    col = 0

    for i in range(Nx_basis):

        Px0 = Px[i]
        Pxd0 = Pxd[i]
        Pxdd0 = Pxdd[i]

        for j in range(Ny_basis):

            Py0 = Py[j]
            Pyd0 = Pyd[j]
            Pydd0 = Pydd[j]

            Pxy = Px0 * Py0

            for k_ in range(Nt_basis):

                Pt0 = Pt[k_]
                Ptd0 = Ptd[k_]

                # u_t
                Dt_col = Pxy * Ptd0

                # u_x
                Dx_col = Pxd0 * Py0 * Pt0

                # u_y
                Dy_col = Px0 * Pyd0 * Pt0

                # u_xx
                Dxx_col = Pxdd0 * Py0 * Pt0

                # u_yy
                Dyy_col = Px0 * Pydd0 * Pt0

                # PDE:
                # u_t + ax_vel*u_x + ay_vel*u_y
                # - k*(u_xx + u_yy)

                A_pde[:, col] = (
                    Dt_col
                    + ax_vel * Dx_col
                    + ay_vel * Dy_col
                    - k * (Dxx_col + Dyy_col)
                )

                col += 1

    # Release large temporary Legendre arrays
    del Px, Pxd, Pxdd
    del Py, Pyd, Pydd
    del Pt, Ptd

    return A_pde


# ==========================================================
# Tensor-product Chebyshev collocation points
# ==========================================================

Nx_col = 46
Ny_col = 46
Nt_col = 21

ix = np.arange(Nx_col)
iy = np.arange(Ny_col)
it = np.arange(Nt_col)

x_cheb = 0.5 * Lx * (
    np.cos(
        (2 * ix + 1) * np.pi / (2 * Nx_col)
    ) + 1.0
)

y_cheb = 0.5 * Ly * (
    np.cos(
        (2 * iy + 1) * np.pi / (2 * Ny_col)
    ) + 1.0
)

t_cheb = 0.5 * T * (
    np.cos(
        (2 * it + 1) * np.pi / (2 * Nt_col)
    ) + 1.0
)


# ==========================================================
# PDE temporal points
#
# Original Chebyshev-Gauss points + t = 1
# ==========================================================

t_pde = np.concatenate([
    t_cheb,
    np.array([T])
])

Xc, Yc, Tc = np.meshgrid(
    x_cheb,
    y_cheb,
    t_pde,
    indexing="ij"
)

x_int = Xc.ravel()
y_int = Yc.ravel()
t_int = Tc.ravel()

print(
    f"PDE collocation points : {len(x_int)}"
)


# ==========================================================
# PDE operator
# ==========================================================

A_pde = pde_operator_matrix_3d(
    x_int,
    y_int,
    t_int
)

b_pde = np.zeros(
    len(x_int)
)


# ==========================================================
# Initial condition
# ==========================================================

Nx_ic = 111
Ny_ic = 111

ix_ic = np.arange(Nx_ic)
iy_ic = np.arange(Ny_ic)

x_ic_cheb = 0.5 * Lx * (
    np.cos(
        (2 * ix_ic + 1) * np.pi / (2 * Nx_ic)
    ) + 1.0
)

y_ic_cheb = 0.5 * Ly * (
    np.cos(
        (2 * iy_ic + 1) * np.pi / (2 * Ny_ic)
    ) + 1.0
)

Xi, Yi = np.meshgrid(
    x_ic_cheb,
    y_ic_cheb,
    indexing="ij"
)

x_ic = Xi.ravel()
y_ic = Yi.ravel()
t_ic = np.zeros_like(x_ic)

P_ic = basis_matrix_3d(
    x_ic,
    y_ic,
    t_ic
)

u_ic = u_exact(
    x_ic,
    y_ic,
    t_ic
)

print(
    f"Initial condition points : {len(x_ic)}"
)


# ==========================================================
# Spatial BC temporal points
#
# Original Chebyshev-Gauss points
# + t = 0
# + t = 1
# ==========================================================

t_bc = np.concatenate([
    np.array([0.0]),
    t_cheb,
    np.array([T])
])


# ==========================================================
# Left & Right boundaries
# ==========================================================

Yb, Tb = np.meshgrid(
    y_cheb,
    t_bc,
    indexing="ij"
)

x_left = np.zeros(
    Yb.size
)

x_right = Lx * np.ones(
    Yb.size
)

y_side = Yb.ravel()
t_side = Tb.ravel()

P_left = basis_matrix_3d(
    x_left,
    y_side,
    t_side
)

P_right = basis_matrix_3d(
    x_right,
    y_side,
    t_side
)

u_left = u_exact(
    x_left,
    y_side,
    t_side
)

u_right = u_exact(
    x_right,
    y_side,
    t_side
)


# ==========================================================
# Bottom & Top boundaries
# ==========================================================

Xb, Tb2 = np.meshgrid(
    x_cheb,
    t_bc,
    indexing="ij"
)

x_side = Xb.ravel()

y_bottom = np.zeros(
    Xb.size
)

y_top = Ly * np.ones(
    Xb.size
)

t_side2 = Tb2.ravel()

P_bottom = basis_matrix_3d(
    x_side,
    y_bottom,
    t_side2
)

P_top = basis_matrix_3d(
    x_side,
    y_top,
    t_side2
)

u_bottom = u_exact(
    x_side,
    y_bottom,
    t_side2
)

u_top = u_exact(
    x_side,
    y_top,
    t_side2
)


# ==========================================================
# Assemble weighted least-squares system
# ==========================================================

w_ic = 1.0
w_bc = 1.0

A = np.vstack([
    A_pde,
    w_ic * P_ic,
    w_bc * P_left,
    w_bc * P_right,
    w_bc * P_bottom,
    w_bc * P_top
])

b = np.concatenate([
    b_pde,
    w_ic * u_ic,
    w_bc * u_left,
    w_bc * u_right,
    w_bc * u_bottom,
    w_bc * u_top
])

print("=" * 60)
print("Linear system")
print("=" * 60)

print(
    "Matrix shape :",
    A.shape
)


# ==========================================================
# Solve
# ==========================================================

beta = scipy.linalg.lstsq(
    A,
    b,
    lapack_driver="gelsy"
)[0]

np.save(
    "P5_beta.npy",
    beta
)

solve_time = time.time() - t0

print(
    f"Solve time : {solve_time:.2f} sec"
)


# ==========================================================
# Residual
# ==========================================================

residual = A @ beta - b

residual_norm = np.linalg.norm(
    residual
)


# ==========================================================
# Interior / PDE-point error
# ==========================================================

Phi_int = basis_matrix_3d(
    x_int,
    y_int,
    t_int
)

U_pred = Phi_int @ beta

U_exact = u_exact(
    x_int,
    y_int,
    t_int
)

error = U_pred - U_exact

rms_error = np.sqrt(
    np.mean(
        error**2
    )
)

relative_l2 = (
    np.linalg.norm(error)
    /
    np.linalg.norm(U_exact)
)

linf_error = np.max(
    np.abs(error)
)


# ==========================================================
# Boundary error
# ==========================================================

boundary_pred = np.concatenate([
    P_left @ beta,
    P_right @ beta,
    P_bottom @ beta,
    P_top @ beta
])

boundary_exact = np.concatenate([
    u_left,
    u_right,
    u_bottom,
    u_top
])

boundary_rms = np.sqrt(
    np.mean(
        (
            boundary_pred
            -
            boundary_exact
        )**2
    )
)


# ==========================================================
# Training statistics
# ==========================================================

print("=" * 60)
print("Training-point statistics")
print("=" * 60)

print(
    f"Residual norm            : "
    f"{residual_norm:.2e}"
)

print(
    f"Interior RMS Error       : "
    f"{rms_error:.2e}"
)

print(
    f"Interior Relative L2     : "
    f"{relative_l2:.2e}"
)

print(
    f"Interior Linf Error      : "
    f"{linf_error:.2e}"
)

print(
    f"Boundary RMS Error       : "
    f"{boundary_rms:.2e}"
)

print("=" * 60)


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
    indexing="ij"
)


# ==========================================================
# Time levels for evaluation
# ==========================================================

plot_times = [
    0.00,
    0.20,
    0.40,
    0.60,
    0.80,
    1.00
]


# ==========================================================
# Storage
# ==========================================================

X_all = []
Y_all = []
T_all = []
U_all = []

print("=" * 60)
print("Evaluation on test grid")
print("=" * 60)


# ==========================================================
# Evaluate MELM solution
# ==========================================================

for tp in plot_times:

    print(
        f"Evaluating t = {tp:.2f}"
    )

    x_plot = X_test.ravel()

    y_plot = Y_test.ravel()

    t_plot = np.full_like(
        x_plot,
        tp
    )

    Phi = basis_matrix_3d(
        x_plot,
        y_plot,
        t_plot
    )

    U_pred = Phi @ beta

    U_exact = u_exact(
        x_plot,
        y_plot,
        t_plot
    )

    error = U_pred - U_exact

    rms = np.sqrt(
        np.mean(
            error**2
        )
    )

    rel_l2 = (
        np.linalg.norm(error)
        /
        np.linalg.norm(U_exact)
    )

    linf = np.max(
        np.abs(error)
    )

    print(
        f"   RMS Error         = "
        f"{rms:.2e}"
    )

    print(
        f"   Relative L2 Error = "
        f"{rel_l2:.2e}"
    )

    print(
        f"   Linf Error        = "
        f"{linf:.2e}"
    )

    X_all.append(
        x_plot
    )

    Y_all.append(
        y_plot
    )

    T_all.append(
        t_plot
    )

    U_all.append(
        U_pred
    )


# ==========================================================
# Merge all time levels
# ==========================================================

X_all = np.concatenate(
    X_all
)

Y_all = np.concatenate(
    Y_all
)

T_all = np.concatenate(
    T_all
)

U_all = np.concatenate(
    U_all
)


# ==========================================================
# Overall error over ALL test points
#
# 101 x 101 x 6 = 61,206 points
# ==========================================================

U_exact_all = u_exact(
    X_all,
    Y_all,
    T_all
)

error_all = U_all - U_exact_all

rms_all = np.sqrt(
    np.mean(
        error_all**2
    )
)

relative_l2_all = (
    np.linalg.norm(error_all)
    /
    np.linalg.norm(U_exact_all)
)

linf_all = np.max(
    np.abs(error_all)
)


print("=" * 60)
print("Overall test-grid statistics")
print("=" * 60)

print(
    f"Total test points       : "
    f"{len(U_all)}"
)

print(
    f"RMS Error               : "
    f"{rms_all:.2e}"
)

print(
    f"Relative L2 Error       : "
    f"{relative_l2_all:.2e}"
)

print(
    f"Linf Error              : "
    f"{linf_all:.2e}"
)

print("=" * 60)


# ==========================================================
# Save CSV
# ==========================================================

data = np.column_stack(
    (
        X_all,
        Y_all,
        T_all,
        U_all
    )
)

np.savetxt(
    "P5_CT_MELM.csv",
    data,
    delimiter=",",
    header="X,Y,T,U",
    comments=""
)

print("=" * 60)
print("Saved solution")
print("=" * 60)

print(
    "Filename : P5_CT_MELM.csv"
)

print(
    f"Total stored points : "
    f"{len(U_all)}"
)

print("=" * 60)

