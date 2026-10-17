#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter


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
T = 1.0

k = 0.01

U = 0.8
Pe = U * Lx / k

print(f"Peclet number = {Pe:.2f}")


# ==========================================================
# Exact solution
# ==========================================================

ax_vel = 0.8
ay_vel = 0.8


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
# Load TEST data exactly as provided
# ==========================================================

df = pd.read_csv("data.csv")


# ==========================================================
# Check required columns
# ==========================================================

required_columns = [
    "x",
    "y",
    "t0020",
    "t0040",
    "t0060",
    "t0080",
    "t0100"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in data.csv: {missing_columns}"
    )


# ==========================================================
# Sort the existing TEST data
#
# No new test points are generated.
# The coordinates are used exactly as supplied.
# ==========================================================

df = df.sort_values(
    by=["x", "y"]
).reset_index(drop=True)


# ==========================================================
# Existing test coordinates
# ==========================================================

x_test = df["x"].unique()
y_test = df["y"].unique()

Nx_test = len(x_test)
Ny_test = len(y_test)

print(f"Nx = {Nx_test}")
print(f"Ny = {Ny_test}")
print(f"Number of test points = {len(df)}")


# ==========================================================
# Verify that the data contains a complete rectangular grid
# ==========================================================

if len(df) != Nx_test * Ny_test:

    raise ValueError(
        "The CSV does not contain a complete rectangular "
        "x-y test grid."
    )


# ==========================================================
# Use the coordinates AS THEY ARE in the test data
# ==========================================================

X_test, Y_test = np.meshgrid(
    x_test,
    y_test,
    indexing="ij"
)


# ==========================================================
# Selected time levels
# ==========================================================

plot_times = [
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
]


# ==========================================================
# Correct DT-PINN output correspondence
#
# dt = 0.001 s
# Output stored every 10 time steps
# Output interval = 0.01 s
#
# t0001 -> 0.01 s
# t0020 -> 0.20 s
# t0040 -> 0.40 s
# t0060 -> 0.60 s
# t0080 -> 0.80 s
# t0100 -> 1.00 s
# ==========================================================

time_columns = {
    0.2: "t0020",
    0.4: "t0040",
    0.6: "t0060",
    0.8: "t0080",
    1.0: "t0100"
}


# ==========================================================
# Load numerical solution and calculate exact solution
# ==========================================================

exact_profiles = {}
solution_profiles = {}

for tp in plot_times:

    column = time_columns[tp]

    # ------------------------------------------------------
    # Numerical DT-PINN solution
    # ------------------------------------------------------

    Un = df[column].to_numpy().reshape(
        Nx_test,
        Ny_test
    )

    # ------------------------------------------------------
    # Exact solution evaluated at the SAME test coordinates
    # ------------------------------------------------------

    Ue = u_exact(
        X_test,
        Y_test,
        tp
    )

    exact_profiles[tp] = Ue
    solution_profiles[tp] = Un

    # ------------------------------------------------------
    # Error
    # ------------------------------------------------------

    Err = np.abs(Ue - Un)

    print(
        f"t={tp:.2f} s | "
        f"column={column} | "
        f"RMS={np.sqrt(np.mean(Err**2)):.3e} | "
        f"Linf={np.max(Err):.3e}"
    )


# ==========================================================
# Common color scale for Exact and DT-PINN
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
    np.min(U)
    for U in all_values
)

vmax = max(
    np.max(U)
    for U in all_values
)


levels_solution = np.linspace(
    vmin,
    vmax,
    60
)


# ==========================================================
# Figure
# ==========================================================

fig = plt.figure(
    figsize=(26, 6 * len(plot_times)),
    constrained_layout=True
)


# ==========================================================
# Plot each time level
# ==========================================================

for r, tp in enumerate(plot_times):

    Ue = exact_profiles[tp]
    Un = solution_profiles[tp]

    Err = np.abs(Ue - Un)

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
        f"Exact ($t={tp:.2f}$)"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xticks(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )

    ax.set_yticks(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )

    ax.set_aspect("equal")


    # ======================================================
    # DT-PINN contour
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
        f"DT-PINN ($t={tp:.2f}$)"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xticks(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )

    ax.set_yticks(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )

    ax.set_aspect("equal")

    cb = fig.colorbar(
        c2,
        ax=ax,
        shrink=0.68,
        pad=0.02
    )

    format_colorbar(cb)


    # ======================================================
    # Absolute error contour
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
        f"|Exact-PINN| ($t={tp:.2f}$)"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xticks(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )

    ax.set_yticks(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )

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

    ax.plot_surface(
        X_test,
        Y_test,
        Ue,
        cmap="jet",
        edgecolor="none"
    )

    ax.set_title(
        f"Exact ($t={tp:.2f}$)"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xticks(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )

    ax.set_yticks(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )

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
    # DT-PINN surface
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

    ax.set_title(
        f"PINN ($t={tp:.2f}$)"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_zlabel(
        r"$\phi$",
        fontsize=14
    )

    ax.set_xticks(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )

    ax.set_yticks(
        [0.0, 0.5, 1.0, 1.5, 2.0]
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

    cb = fig.colorbar(
        surf,
        ax=ax,
        shrink=0.55,
        pad=0.02
    )

    format_colorbar(cb)


# ==========================================================
# Save only — do not display
# ==========================================================

plt.savefig(
    "P5_DT_PINN.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.close(fig)


# In[ ]:




