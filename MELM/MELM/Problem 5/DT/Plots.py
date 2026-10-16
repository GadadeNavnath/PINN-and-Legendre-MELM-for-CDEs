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
# Load stored solution
# ==========================================================

df = pd.read_csv("P5_DT_MELM.csv")

error_df = pd.read_csv("P5_DT_MELM_errors.csv")


# ==========================================================
# ERROR VS TIME
# ==========================================================

time_history = error_df["Time"].values
rel_l2_history = error_df["Relative_L2_Error"].values
linf_history = error_df["Max_Absolute_Error"].values


fig_error = plt.figure(
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

plt.ylim(1e-5, 1e-3)

plt.yticks(
    [1e-5, 1e-4, 1e-3],
    [r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$"],
    fontsize=11
)

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.legend(
    fontsize=11
)

plt.tight_layout()

plt.savefig(
    "P5_DT_MELM_error_vs_time.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()


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
# Time levels
# ==========================================================

plot_times = [0.2, 0.4, 0.6, 0.8, 1.0]

exact_profiles = {}
solution_profiles = {}


# ==========================================================
# Load solutions
# ==========================================================

for tp in plot_times:

    Ue = u_exact(
        X_test.ravel(),
        Y_test.ravel(),
        tp * np.ones(X_test.size)
    ).reshape(Nx_test, Ny_test)

    temp = df[np.isclose(df["T"], tp)]

    Un = temp["U"].values.reshape(
        Nx_test,
        Ny_test
    )

    exact_profiles[tp] = Ue
    solution_profiles[tp] = Un

    err = np.abs(Ue - Un)

    print(
        f"t={tp:.2f}   "
        f"RMS={np.sqrt(np.mean(err**2)):.3e}   "
        f"Linf={np.max(err):.3e}"
    )


# ==========================================================
# Common color scale
# ==========================================================

all_values = []

for tp in plot_times:

    all_values.append(exact_profiles[tp])
    all_values.append(solution_profiles[tp])

vmin = min(U.min() for U in all_values)
vmax = max(U.max() for U in all_values)

levels_solution = np.linspace(
    vmin,
    vmax,
    60
)


# ==========================================================
# Plot all time levels
# ==========================================================

fig = plt.figure(
    figsize=(26, 6 * len(plot_times)),
    constrained_layout=True
)


for r, tp in enumerate(plot_times):

    Ue = exact_profiles[tp]
    Un = solution_profiles[tp]

    Err = np.abs(Ue - Un)

    err_max = Err.max()


    # =====================================================
    # Exact contour
    # =====================================================

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

    ax.set_title(f"Exact (t={tp:.2f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xticks(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )

    ax.set_yticks(
        [0.0, 0.5, 1.0, 1.5, 2.0]
    )

    ax.set_aspect("equal")


    # =====================================================
    # MELM contour
    # =====================================================

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

    ax.set_title(f"MELM (t={tp:.2f})")
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


    # =====================================================
    # Error contour
    # =====================================================

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

    ax.set_title("|Exact-MELM|")
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


    # =====================================================
    # Exact surface
    # =====================================================

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


    # =====================================================
    # MELM surface
    # =====================================================

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
# Save final multi-time-level figure
# ==========================================================

plt.savefig(
    "P5_DT_MELM.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()


# In[ ]:




