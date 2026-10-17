#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter


# ==========================================================
# Plot settings
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
# Time levels and files
# ==========================================================

plot_times = [
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
]

file_names = [
    "data/t_0020.csv",
    "data/t_0040.csv",
    "data/t_0060.csv",
    "data/t_0080.csv",
    "data/t_0100.csv"
]


# ==========================================================
# Storage
# ==========================================================

X_profiles = {}
Y_profiles = {}

exact_profiles = {}
pinn_profiles = {}


# ==========================================================
# Load CSV files
# ==========================================================

for tp, filename in zip(plot_times, file_names):

    df = pd.read_csv(filename)

    # Sort according to stored coordinates
    df = df.sort_values(
        by=["x", "y"]
    ).reset_index(drop=True)

    # Coordinates directly from CSV
    x_unique = np.sort(df["x"].unique())
    y_unique = np.sort(df["y"].unique())

    Nx = len(x_unique)
    Ny = len(y_unique)

    # Mesh from stored coordinates
    X, Y = np.meshgrid(
        x_unique,
        y_unique,
        indexing="ij"
    )

    # Solutions
    Ue = df["Exact"].values.reshape(Nx, Ny)
    Up = df["PINN"].values.reshape(Nx, Ny)

    # Store
    X_profiles[tp] = X
    Y_profiles[tp] = Y

    exact_profiles[tp] = Ue
    pinn_profiles[tp] = Up

    # ======================================================
    # PINN error
    # ======================================================

    Err = np.abs(Ue - Up)

    RMS = np.sqrt(
        np.mean(Err**2)
    )

    Linf = np.max(Err)

    print(
        f"t={tp:.2f}   "
        f"RMS={RMS:.3e}   "
        f"Linf={Linf:.3e}"
    )
# ==========================================================
# Common color scale for Exact and PINN
# ==========================================================

all_solution_values = []

for tp in plot_times:

    all_solution_values.append(
        exact_profiles[tp]
    )

    all_solution_values.append(
        pinn_profiles[tp]
    )


vmin = min(
    U.min()
    for U in all_solution_values
)

vmax = max(
    U.max()
    for U in all_solution_values
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

    X = X_profiles[tp]
    Y = Y_profiles[tp]

    Ue = exact_profiles[tp]
    Up = pinn_profiles[tp]

    Err = np.abs(
        Ue - Up
    )

    # ======================================================
    # Error scale
    #
    # Separate scale for each time level so that small
    # PINN errors remain visible.
    # ======================================================

    err_max = np.max(Err)

    if err_max == 0:
        err_max = 1.0e-16

    levels_error = np.linspace(
        0,
        err_max,
        40
    )


    # ======================================================
    # Common z-axis scale
    # ======================================================

    zmin = min(
        Ue.min(),
        Up.min()
    )

    zmax = max(
        Ue.max(),
        Up.max()
    )


    # ======================================================
    # 1. Exact contour
    # ======================================================

    ax = fig.add_subplot(
        len(plot_times),
        5,
        5 * r + 1
    )

    c1 = ax.contourf(
        X,
        Y,
        Ue,
        levels=levels_solution,
        cmap="jet"
    )

    ax.set_title(
        f"Exact (t={tp:.2f})"
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
    # 2. PINN contour
    # ======================================================

    ax = fig.add_subplot(
        len(plot_times),
        5,
        5 * r + 2
    )

    c2 = ax.contourf(
        X,
        Y,
        Up,
        levels=levels_solution,
        cmap="jet"
    )

    ax.set_title(
        f"PINN (t={tp:.2f})"
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
    # 3. Absolute error contour
    # ======================================================

    ax = fig.add_subplot(
        len(plot_times),
        5,
        5 * r + 3
    )

    c3 = ax.contourf(
        X,
        Y,
        Err,
        levels=levels_error,
        cmap="jet"
    )

    ax.set_title(
        r"$|\mathrm{Exact}-\mathrm{PINN}|$"
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
    # 4. Exact surface
    # ======================================================

    ax = fig.add_subplot(
        len(plot_times),
        5,
        5 * r + 4,
        projection="3d"
    )

    surf = ax.plot_surface(
        X,
        Y,
        Ue,
        cmap="jet",
        edgecolor="none"
    )

    ax.set_title("Exact")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

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


    # ======================================================
    # 5. PINN surface
    # ======================================================

    ax = fig.add_subplot(
        len(plot_times),
        5,
        5 * r + 5,
        projection="3d"
    )

    surf = ax.plot_surface(
        X,
        Y,
        Up,
        cmap="jet",
        edgecolor="none"
    )

    ax.set_title("PINN")

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
# Save
# ==========================================================

plt.savefig(
    "P5_CT_PINN.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()


# In[ ]:




