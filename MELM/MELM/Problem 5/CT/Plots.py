#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter


# ==========================================================
# Plot parameters
# ==========================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 20,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# ==========================================================
# Colorbar formatting
# ==========================================================

def format_colorbar(cb):

    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    cb.formatter = formatter
    cb.locator = mticker.MaxNLocator(6)
    cb.update_ticks()

    cb.ax.tick_params(labelsize=14)

    cb.ax.yaxis.get_offset_text().set_size(14)


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
# Velocity field
# ==========================================================

ax_vel = 0.8
ay_vel = 0.8


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
# Load stored MELM solution
# ==========================================================

df = pd.read_csv("P5_CT_MELM.csv")


# ==========================================================
# Test grid
# ==========================================================

Nx_test = 101
Ny_test = 101

x_test = np.linspace(0, Lx, Nx_test)
y_test = np.linspace(0, Ly, Ny_test)

X_test, Y_test = np.meshgrid(
    x_test,
    y_test,
    indexing="ij"
)


# ==========================================================
# Time levels for error calculation
# ==========================================================

times = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
]


# ==========================================================
# Time levels for plotting
# ==========================================================

plot_times = [
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
]


# ==========================================================
# Storage
# ==========================================================

all_exact = []
all_melm = []

exact_profiles = {}
solution_profiles = {}


# ==========================================================
# ==========================================================
# ERROR CALCULATION
# ==========================================================
# ==========================================================

print()
print("=" * 100)
print("P5_CT_MELM : Solution Magnitudes and Errors")
print("=" * 100)

print(f"Total stored points : {len(df)}")

print()
print(
    f"{'t':>6}"
    f"{'||u_exact||_2':>18}"
    f"{'||u_MELM||_2':>18}"
    f"{'||error||_2':>18}"
    f"{'Relative L2':>18}"
    f"{'Linf':>18}"
    f"{'RMS':>18}"
)

print("-" * 114)


for tp in times:

    # ------------------------------------------------------
    # Exact solution
    # ------------------------------------------------------

    Ue = u_exact(
        X_test,
        Y_test,
        tp
    )

    # ------------------------------------------------------
    # Select MELM solution from CSV
    # ------------------------------------------------------

    temp = df[
        np.isclose(
            df["T"],
            tp
        )
    ]

    Un = temp["U"].values

    if len(Un) != Nx_test * Ny_test:

        raise ValueError(
            f"At t={tp:.2f}: expected "
            f"{Nx_test * Ny_test} points, "
            f"found {len(Un)}."
        )

    Un = Un.reshape(
        Nx_test,
        Ny_test
    )

    # ------------------------------------------------------
    # Error
    # ------------------------------------------------------

    error = Ue - Un

    # ------------------------------------------------------
    # Norms
    # ------------------------------------------------------

    exact_l2 = np.linalg.norm(Ue)

    melm_l2 = np.linalg.norm(Un)

    error_l2 = np.linalg.norm(error)

    relative_l2 = (
        error_l2 / exact_l2
    )

    linf = np.max(
        np.abs(error)
    )

    rms = np.sqrt(
        np.mean(error ** 2)
    )

    # ------------------------------------------------------
    # Store for global space-time error
    # ------------------------------------------------------

    all_exact.append(
        Ue.ravel()
    )

    all_melm.append(
        Un.ravel()
    )

    # ------------------------------------------------------
    # Store profiles for plotting
    # ------------------------------------------------------

    if tp in plot_times:

        exact_profiles[tp] = Ue
        solution_profiles[tp] = Un

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    print(
        f"{tp:6.2f}"
        f"{exact_l2:18.6e}"
        f"{melm_l2:18.6e}"
        f"{error_l2:18.6e}"
        f"{relative_l2:18.6e}"
        f"{linf:18.6e}"
        f"{rms:18.6e}"
    )


# ==========================================================
# Overall space-time error
# ==========================================================

all_exact = np.concatenate(
    all_exact
)

all_melm = np.concatenate(
    all_melm
)

all_error = (
    all_exact - all_melm
)


global_exact_l2 = np.linalg.norm(
    all_exact
)

global_melm_l2 = np.linalg.norm(
    all_melm
)

global_error_l2 = np.linalg.norm(
    all_error
)

global_relative_l2 = (
    global_error_l2 / global_exact_l2
)

global_linf = np.max(
    np.abs(all_error)
)

global_rms = np.sqrt(
    np.mean(all_error ** 2)
)


# ==========================================================
# Overall results
# ==========================================================

print()
print("=" * 70)
print("Overall Space-Time Error")
print("=" * 70)

print(
    f"||u_exact||_2       = "
    f"{global_exact_l2:.2e}"
)

print(
    f"||u_MELM||_2        = "
    f"{global_melm_l2:.2e}"
)

print(
    f"||error||_2         = "
    f"{global_error_l2:.2e}"
)

print(
    f"Relative L2 Error   = "
    f"{global_relative_l2:.2e}"
)

print(
    f"Linf Error          = "
    f"{global_linf:.2e}"
)

print(
    f"RMS Error           = "
    f"{global_rms:.2e}"
)

print("=" * 70)


# ==========================================================
# Common color scale for plots
# ==========================================================

all_values = []

for tp in plot_times:

    all_values.append(
        exact_profiles[tp]
    )

    all_values.append(
        solution_profiles[tp]
    )


vmin = min(
    U.min()
    for U in all_values
)

vmax = max(
    U.max()
    for U in all_values
)

levels_solution = np.linspace(
    vmin,
    vmax,
    60
)


# ==========================================================
# Plot
# ==========================================================

fig = plt.figure(
    figsize=(26, 6 * len(plot_times)),
    constrained_layout=True
)


for r, tp in enumerate(plot_times):

    Ue = exact_profiles[tp]

    Un = solution_profiles[tp]

    Err = np.abs(
        Ue - Un
    )

    err_max = Err.max()


    # ======================================================
    # Exact contour
    # ======================================================

    ax = fig.add_subplot(
        len(plot_times),
        5,
        5 * r + 1
    )

    c1 = ax.contourf(
        X_test,
        Y_test,
        Ue,
        levels=levels_solution,
        cmap="jet"
    )

    ax.set_title(
        f"Exact (t={tp:.2f})"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xticks([
        0.0,
        0.5,
        1.0,
        1.5,
        2.0
    ])

    ax.set_yticks([
        0.0,
        0.5,
        1.0,
        1.5,
        2.0
    ])

    ax.set_aspect("equal")


    # ======================================================
    # MELM contour
    # ======================================================

    ax = fig.add_subplot(
        len(plot_times),
        5,
        5 * r + 2
    )

    c2 = ax.contourf(
        X_test,
        Y_test,
        Un,
        levels=levels_solution,
        cmap="jet"
    )

    ax.set_title(
        f"MELM (t={tp:.2f})"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xticks([
        0.0,
        0.5,
        1.0,
        1.5,
        2.0
    ])

    ax.set_yticks([
        0.0,
        0.5,
        1.0,
        1.5,
        2.0
    ])

    ax.set_aspect("equal")


    cb = fig.colorbar(
        c2,
        ax=ax,
        shrink=0.68,
        pad=0.02
    )

    format_colorbar(cb)


    # ======================================================
    # Error contour
    # ======================================================

    ax = fig.add_subplot(
        len(plot_times),
        5,
        5 * r + 3
    )

    c3 = ax.contourf(
        X_test,
        Y_test,
        Err,
        levels=np.linspace(
            0,
            err_max,
            40
        ),
        cmap="jet"
    )

    ax.set_title(
        "|Exact-MELM|"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xticks([
        0.0,
        0.5,
        1.0,
        1.5,
        2.0
    ])

    ax.set_yticks([
        0.0,
        0.5,
        1.0,
        1.5,
        2.0
    ])

    ax.set_aspect("equal")


    cb = fig.colorbar(
        c3,
        ax=ax,
        shrink=0.68,
        pad=0.02
    )

    format_colorbar(cb)


    # ======================================================
    # Exact surface
    # ======================================================

    ax = fig.add_subplot(
        len(plot_times),
        5,
        5 * r + 4,
        projection="3d"
    )

    surf = ax.plot_surface(
        X_test,
        Y_test,
        Ue,
        cmap="jet",
        edgecolor="none"
    )

    ax.set_title("Exact")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xticks([
        0.0,
        0.5,
        1.0,
        1.5,
        2.0
    ])

    ax.set_yticks([
        0.0,
        0.5,
        1.0,
        1.5,
        2.0
    ])

    zmin = min(
        Ue.min(),
        Un.min()
    )

    zmax = max(
        Ue.max(),
        Un.max()
    )

    ax.set_zlim(
        zmin,
        1.15 * zmax
    )

    ax.set_box_aspect(
        (1, 1, 1)
    )

    ax.view_init(
        elev=28,
        azim=-60
    )


    # ======================================================
    # MELM surface
    # ======================================================

    ax = fig.add_subplot(
        len(plot_times),
        5,
        5 * r + 5,
        projection="3d"
    )

    surf = ax.plot_surface(
        X_test,
        Y_test,
        Un,
        cmap="jet",
        edgecolor="none"
    )

    ax.set_title("MELM")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(
        r"$\phi$",
        fontsize=14
    )

    ax.set_xticks([
        0.0,
        0.5,
        1.0,
        1.5,
        2.0
    ])

    ax.set_yticks([
        0.0,
        0.5,
        1.0,
        1.5,
        2.0
    ])

    ax.set_zlim(
        zmin,
        1.15 * zmax
    )

    ax.set_box_aspect(
        (1, 1, 1)
    )

    ax.view_init(
        elev=28,
        azim=-60
    )


    cb = fig.colorbar(
        surf,
        ax=ax,
        shrink=0.55,
        pad=0.02
    )

    format_colorbar(cb)


# ==========================================================
# Save figure
# ==========================================================

plt.savefig(
    "P5_CT_MELM.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()


# In[ ]:




