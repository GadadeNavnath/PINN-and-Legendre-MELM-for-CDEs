#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numpy.polynomial.legendre import legval, legder
import matplotlib.pyplot as plt
import time
from scipy.linalg import lu_factor, lu_solve
from matplotlib.ticker import ScalarFormatter

# =====================================================
# Start timer
# =====================================================

start_time = time.time()

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
# Time discretization
# =====================================================

dt = 0.01

Nt = int(round(T / dt))

print("==============================================")
print("DT-MELM: BDF1 + BDF2")
print("==============================================")

print(f"Domain        : [0, {Lx}]")
print(f"Final time    : {T}")
print(f"Time step     : {dt}")
print(f"Time steps    : {Nt}")
print(f"Convection a  : {a}")
print(f"Diffusion k   : {k}")

# =====================================================
# Legendre basis
# =====================================================
Nx_basis = 60

N_unknown = Nx_basis

print(f"Basis functions : {N_unknown}")

# ====================================================
# Chebyshev-Lobatto collocation points
# =====================================================

Nx_col = Nx_basis

j = np.arange(Nx_col)

xi_cheb = np.cos(
    np.pi * j / (Nx_col - 1)
)

# Map [-1,1] to [0,Lx]

x_cheb = 0.5 * Lx * (
    1.0 + xi_cheb
)

# Put points in ascending order

x_cheb = x_cheb[::-1]

# =====================================================
# Interior training points
# =====================================================

x_int = x_cheb[1:-1]

print(f"Total collocation points : {len(x_cheb)}")
print(f"Interior training points : {len(x_int)}")
print(f"Left boundary            : {x_cheb[0]:.2f}")
print(f"Right boundary           : {x_cheb[-1]:.2f}")

# =====================================================
# Legendre coefficients
# =====================================================

basis_coeffs = []
d1_coeffs = []
d2_coeffs = []

for n in range(Nx_basis):

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

def basis_matrix(x):

    x = np.asarray(x)

    xh = (
        2.0 * x / Lx
        -
        1.0
    )

    cols = []

    for i in range(Nx_basis):

        Px = legval(
            xh,
            basis_coeffs[i]
        )

        cols.append(Px)

    return np.vstack(cols).T

# =====================================================
# First derivative matrix
# =====================================================

def derivative_matrix(x):

    x = np.asarray(x)

    xh = (
        2.0 * x / Lx
        -
        1.0
    )

    sx = 2.0 / Lx

    cols = []

    for i in range(Nx_basis):

        dPx = (
            sx
            *
            legval(
                xh,
                d1_coeffs[i]
            )
        )

        cols.append(dPx)

    return np.vstack(cols).T

# =====================================================
# Second derivative matrix
# =====================================================

def second_derivative_matrix(x):

    x = np.asarray(x)

    xh = (
        2.0 * x / Lx
        -
        1.0
    )

    sx = 2.0 / Lx

    cols = []

    for i in range(Nx_basis):

        d2Px = (
            sx**2
            *
            legval(
                xh,
                d2_coeffs[i]
            )
        )

        cols.append(d2Px)

    return np.vstack(cols).T


# =====================================================
# Training matrices
# =====================================================

P = basis_matrix(x_int)

Px = derivative_matrix(x_int)

Pxx = second_derivative_matrix(x_int)

# =====================================================
# Boundary matrices
# =====================================================

P_left = basis_matrix(
    np.array([0.0])
)

P_right = basis_matrix(
    np.array([Lx])
)

# =====================================================
# Test grid
# x_test is NOT used to determine beta.
# It is used only after beta is obtained for
# independent error evaluation.
# =====================================================

Nx_test = 500

x_test = np.linspace(
    0.0,
    Lx,
    Nx_test
)

P_test = basis_matrix(
    x_test
)

# =====================================================
# Solution profiles to store
# =====================================================

plot_times = np.arange(
    0.5,
    T + 0.001,
    0.5
)

solution_profiles = {}

# =====================================================
# Initial condition

u_old = u_exact(
    x_int,
    0.0
)

print("\nInitial condition:")
print("u^0 is taken directly from the prescribed initial condition.")

# =====================================================
# Error history
# =====================================================

time_history = []

linf_history = []

rel_l2_history = []

overall_error_sq = 0.0

overall_exact_sq = 0.0

# =====================================================
# FIRST TIME STEP: BDF1
# =====================================================
# (u^1 - u^0)/dt
# + a*u_x^1
# - k*u_xx^1 = 0

# =====================================================

t = dt

print("\n==============================================")
print("BDF1")
print("==============================================")

print(f"Computing t = {t:.2f}")

# -----------------------------------------------------
# BDF1 interior matrix
# -----------------------------------------------------

A_int = (
    P
    +
    a * dt * Px
    -
    k * dt * Pxx
)

# -----------------------------------------------------
# Add boundary equations
# -----------------------------------------------------

A_bdf1 = np.vstack([
    P_left,
    A_int,
    P_right
])

# -----------------------------------------------------
# BDF1 right-hand side
# -----------------------------------------------------

rhs_bdf1 = np.concatenate([
    [u_exact(0.0, t)],
    u_old,
    [u_exact(Lx, t)]
])

# -----------------------------------------------------
# Solve for beta^1
# -----------------------------------------------------

beta = np.linalg.solve(
    A_bdf1,
    rhs_bdf1
)

# -----------------------------------------------------
# Numerical solution at training points
# -----------------------------------------------------
u_curr = P @ beta

# =====================================================
# Evaluate BDF1 solution on test points
# =====================================================

u_num_test = P_test @ beta

u_true_test = u_exact(
    x_test,
    t
)

error = (
    u_num_test
    -
    u_true_test
)

# =====================================================
# Discrete errors at t = 0.01
# =====================================================

abs_error = np.abs(error)

# L-infinity

linf = np.max(
    abs_error
)


# Discrete L2 norm

error_l2_discrete = np.sqrt(
    np.sum(
        error**2
    )
)

# Discrete exact L2 norm

exact_l2_discrete = np.sqrt(
    np.sum(
        u_true_test**2
    )
)

# Relative discrete L2

rel_l2 = (
    error_l2_discrete
    /
    exact_l2_discrete
)

# =====================================================
# Store BDF1 errors
# =====================================================

time_history.append(t)

linf_history.append(linf)

rel_l2_history.append(rel_l2)

# =====================================================
# Add BDF1 contribution to overall error
# =====================================================

overall_error_sq += np.sum(
    error**2
)

overall_exact_sq += np.sum(
    u_true_test**2
)

# =====================================================
# Store profile if required
# =====================================================

if np.any(
    np.isclose(
        t,
        plot_times,
        atol=1e-12
    )
):

    solution_profiles[
        round(t, 2)
    ] = u_num_test.copy()

# =====================================================
# Prepare BDF2
# =====================================================
#
# At this point:
#
# u_prev = u^0
# u_curr = u^1
#
# =====================================================

u_prev = u_old.copy()

u_curr = u_curr.copy()


# =====================================================
# BDF2 matrix
# =====================================================
#
# BDF2:
#
# (3u^{n+1} - 4u^n + u^{n-1})/(2dt)
# + a*u_x^{n+1}
# - k*u_xx^{n+1}
# = 0
#
# Dividing by 2:
#
# 1.5*u^{n+1}
# + dt*a*u_x^{n+1}
# - dt*k*u_xx^{n+1}
#
# =
#
# 2*u^n - 0.5*u^{n-1}
#
# =====================================================

A_int_bdf2 = (
    1.5 * P
    +
    a * dt * Px
    -
    k * dt * Pxx
)

# -----------------------------------------------------
# Add boundary equations
# -----------------------------------------------------

A_bdf2 = np.vstack([
    P_left,
    A_int_bdf2,
    P_right
])

# =====================================================
# LU factorization
# =====================================================
#
# A_bdf2 does not change with time.
#
# Therefore factorize once and reuse it.
#
# =====================================================

LU, piv = lu_factor(
    A_bdf2
)

# =====================================================
# BDF2 TIME INTEGRATION
# =====================================================
#
# First BDF2 step:
#
# n = 1
#
# t = (1+1)*dt = 0.02
#
# =====================================================

print("\n==============================================")
print("BDF2")
print("==============================================")


for n in range(1, Nt):
    t = (
        n + 1
    ) * dt
    # -------------------------------------------------
    # BDF2 interior RHS
    # 2*u^n - 0.5*u^{n-1}
    # -------------------------------------------------
    rhs_int = (
        2.0 * u_curr
        -
        0.5 * u_prev
    )
    # -------------------------------------------------
    # Boundary conditions
    # -------------------------------------------------
    rhs = np.concatenate([
        [u_exact(0.0, t)],
        rhs_int,
        [u_exact(Lx, t)]
    ])
    # -------------------------------------------------
    # Solve BDF2 system
    # -------------------------------------------------
    beta = lu_solve(
        (LU, piv),
        rhs
    )
    # -------------------------------------------------
    # Numerical solution at training points
    # -------------------------------------------------
    u_new = P @ beta
    # =================================================
    # Evaluate on independent test grid
    # =================================================
    u_num_test = P_test @ beta

    u_true_test = u_exact(
        x_test,
        t
    )

    error = (
        u_num_test
        -
        u_true_test
    )
    # =================================================
    # Discrete errors at current time
    # =================================================
    abs_error = np.abs(
        error
    )
    # L-infinity

    linf = np.max(
        abs_error
    )
    # Discrete L2 norm

    error_l2_discrete = np.sqrt(
        np.sum(
            error**2
        )
    )
    # Discrete exact L2 norm

    exact_l2_discrete = np.sqrt(
        np.sum(
            u_true_test**2
        )
    )
    # Relative discrete L2

    rel_l2 = (
        error_l2_discrete
        /
        exact_l2_discrete
    )
    # =================================================
    # Store error history
    # =================================================

    time_history.append(t)

    linf_history.append(linf)

    rel_l2_history.append(rel_l2)


    # =================================================
    # Accumulate overall discrete space-time error
    # =================================================

    overall_error_sq += np.sum(
        error**2
    )

    overall_exact_sq += np.sum(
        u_true_test**2
    )
    # =================================================
    # Store selected solution profiles
    # =================================================

    if np.any(
        np.isclose(
            t,
            plot_times,
            atol=1e-12
        )
    ):

        solution_profiles[
            round(t, 2)
        ] = u_num_test.copy()

   # =================================================
    # Update previous solutions
    # =================================================
    #
    # Before:
    #
    # u_prev = u^{n-1}
    # u_curr = u^n
    #
    # After:
    #
    # u_prev = u^n
    # u_curr = u^{n+1}
    #
    # =================================================

    u_prev = u_curr.copy()

    u_curr = u_new.copy()


# =====================================================
# Stop timer
# =====================================================

end_time = time.time()

total_cpu_time = (
    end_time
    -
    start_time
)

# =====================================================
# Final-time errors
# =====================================================

u_num_final = P_test @ beta

u_exact_final = u_exact(
    x_test,
    T
)

final_error = (
    u_num_final
    -
    u_exact_final
)

# -----------------------------------------------------
# Final Linf
# -----------------------------------------------------

final_linf = np.max(
    np.abs(
        final_error
    )
)
# -----------------------------------------------------
# Final discrete L2
# -----------------------------------------------------

final_l2 = np.sqrt(
    np.sum(
        final_error**2
    )
)
# -----------------------------------------------------
# Final exact discrete L2
# -----------------------------------------------------

final_exact_l2 = np.sqrt(
    np.sum(
        u_exact_final**2
    )
)
# -----------------------------------------------------
# Final relative discrete L2
# -----------------------------------------------------

final_rel_l2 = (
    final_l2
    /
    final_exact_l2
)
# =====================================================
# OVERALL DISCRETE SPACE-TIME RELATIVE L2 ERROR
# =====================================================
#
# This includes:
#
# t = 0.01, 0.02, ..., 5.00
#
# and all Nx_test spatial points.
#
# =====================================================

overall_error_norm = np.sqrt(
    overall_error_sq
)

overall_exact_norm = np.sqrt(
    overall_exact_sq
)

overall_rel_l2 = (
    overall_error_norm
    /
    overall_exact_norm
)

# =====================================================
# Maximum relative L2 error
# =====================================================

max_rel_l2_index = np.argmax(
    rel_l2_history
)

max_rel_l2 = rel_l2_history[
    max_rel_l2_index
]

max_rel_l2_time = time_history[
    max_rel_l2_index
]

# =====================================================
# Print final results
# =====================================================

print("\n==============================================")
print("FINAL RESULTS")
print("==============================================")

print(
    f"CPU time                    = "
    f"{total_cpu_time:.2f} s"
)

print(
    f"Final time                  = "
    f"{T:.2f}"
)

print(
    f"Final Linf error            = "
    f"{final_linf:.2e}"
)

print(
    f"Final discrete L2 error     = "
    f"{final_l2:.2e}"
)

print(
    f"Final Relative L2 error     = "
    f"{final_rel_l2:.2e}"
)

print(
    f"Overall discrete Relative "
    f"L2 error                   = "
    f"{overall_rel_l2:.2e}"
)

print(
    f"Maximum Relative L2 error   = "
    f"{max_rel_l2:.2e}"
)

print(
    f"Maximum Relative L2 at t    = "
    f"{max_rel_l2_time:.2f}"
)


# =====================================================
# Save final solution
# =====================================================

final_data = np.column_stack([
    x_test,
    u_num_final,
    u_exact_final,
    np.abs(final_error)
])

np.savetxt(
    "P4_DT_MELM.csv",
    final_data,
    delimiter=",",
    header="X,U_MELM,U_exact,Absolute_Error",
    comments=""
)

print(
    "\nFinal solution saved to "
    "P4_DT_MELM.csv"
)

# =====================================================
# PLOTTING SECTION
# =====================================================

# -----------------------------------------------------
# Time levels for plotting
# -----------------------------------------------------

plot_times = np.arange(
    0.5,
    T + 0.001,
    0.5
)


# -----------------------------------------------------
# Viridis colors
# -----------------------------------------------------

cmap = plt.get_cmap("viridis")

colors = cmap(
    np.linspace(
        0.0,
        1.0,
        len(plot_times) + 1
    )
)


# =====================================================
# SOLUTION PROFILES
# =====================================================

plt.figure(
    figsize=(9, 7)
)


# -----------------------------------------------------
# Exact solution at t = 0
# -----------------------------------------------------

u_exact_t0 = u_exact(
    x_test,
    0.0
)

plt.plot(
    x_test,
    u_exact_t0,
    color=colors[0],
    linewidth=2.0,
    linestyle="-",
    label=r"Exact, $t=0$"
)


# -----------------------------------------------------
# Exact and MELM solutions
# -----------------------------------------------------

for i, tp in enumerate(plot_times):

    tp = round(
        tp,
        2
    )

    if tp not in solution_profiles:
        continue

    # Exact solution

    u_ex = u_exact(
        x_test,
        tp
    )

    # MELM solution

    u_melm = solution_profiles[
        tp
    ]

    # -------------------------------------------------
    # Exact: solid
    # -------------------------------------------------

    plt.plot(
        x_test,
        u_ex,
        color=colors[i + 1],
        linewidth=1.5,
        linestyle="-",
        label=fr"Exact, $t={tp:g}$"
    )

    # -------------------------------------------------
    # MELM: dashed
    # -------------------------------------------------

    plt.plot(
        x_test,
        u_melm,
        color=colors[i + 1],
        linewidth=1.5,
        linestyle="--",
        label=fr"MELM, $t={tp:g}$"
    )


# -----------------------------------------------------
# Formatting
# -----------------------------------------------------

plt.xlabel(
    r"$x$",
    fontsize=14
)

plt.ylabel(
    r"$\phi(x,t)$",
    fontsize=14
)

plt.xticks(
    fontsize=12
)

plt.yticks(
    fontsize=12
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend(
    ncol=2,
    fontsize=9,
    frameon=True
)

plt.tight_layout()

plt.savefig(
    "P4_DT_MELM_solution.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()


# =====================================================
# ABSOLUTE ERROR PROFILES
# =====================================================

plt.figure(
    figsize=(9, 7)
)

for i, tp in enumerate(plot_times):

    tp = round(
        tp,
        2
    )

    if tp not in solution_profiles:
        continue

    u_melm = solution_profiles[
        tp
    ]

    u_ex = u_exact(
        x_test,
        tp
    )

    error_profile = np.abs(
        u_melm - u_ex
    )

    plt.plot(
        x_test,
        error_profile,
        color=colors[i + 1],
        linewidth=1.5,
        label=fr"$t={tp:g}$"
    )

# -----------------------------------------------------
# Formatting
# -----------------------------------------------------
plt.xlabel(
    r"$x$",
    fontsize=15
)

plt.ylabel(
    "Absolute error",
    fontsize=15
)

plt.xticks(
    fontsize=12
)

plt.yticks(
    fontsize=12
)

# Scientific notation on y-axis
formatter = ScalarFormatter(
    useMathText=True
)

formatter.set_powerlimits(
    (0, 0)
)

plt.gca().yaxis.set_major_formatter(
    formatter
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend(
    ncol=2,
    fontsize=9,
    frameon=True
)

plt.tight_layout()

plt.savefig(
    "P4_DT_MELM_absolute_error.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()


# =====================================================
# ERROR VS TIME
# =====================================================

plt.figure(
    figsize=(9, 7)
)

plt.semilogy(
    time_history,
    linf_history,
    color="black",
    linewidth=1.5,
    label="Maximum absolute error"
)

plt.semilogy(
    time_history,
    rel_l2_history,
    color="red",
    linewidth=1.5,
    label=r"Relative $L_2$ error"
)

plt.xlabel(
    "Time",
    fontsize=14
)

plt.ylabel(
    "Error",
    fontsize=14
)

plt.xticks(
    fontsize=11
)

plt.yticks(
    fontsize=11
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend(
    fontsize=11
)

plt.tight_layout()

plt.savefig(
    "P4_DT_MELM_error_vs_time.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()

