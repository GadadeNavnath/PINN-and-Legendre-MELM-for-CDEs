#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ============================================================
# Problem parameters
# ============================================================

a = 1.0
k = 0.1
dt = 0.01

file_path = "data/data.csv"


# ============================================================
# Exact solution
# ============================================================

def u_exact(x, t):

    return (
        1.0 / np.sqrt(4.0 * t + 1.0)
        * np.exp(
            -((x - 1.0 - a * t) ** 2)
            / (k * (4.0 * t + 1.0))
        )
    )


# ============================================================
# Read PINN data
# ============================================================

df = pd.read_csv(file_path)

time_columns = [
    col for col in df.columns
    if col != "x"
]


# ============================================================
# Determine CSV time values
# ============================================================
#
# t0001 -> 0.01
# t0002 -> 0.02
# ...
# t0500 -> 5.00
#
# ============================================================

csv_times = (
    np.arange(
        1,
        len(time_columns) + 1
    )
    * dt
)


# ============================================================
# Remove duplicate x values if present
# ============================================================

data = (
    df[["x"] + time_columns]
    .groupby(
        "x",
        sort=True
    )
    .mean()
)

x = data.index.to_numpy()

U_pinn = data[
    time_columns
].to_numpy()


# ============================================================
# Requested plotting times
# ============================================================

plot_times = np.arange(
    0.5,
    5.01,
    0.5
)


# ============================================================
# Viridis colors
# ============================================================
#
# Same color convention as IELM:
#
# t = 0       -> colors[0]
# t = 0.5     -> colors[1]
# t = 1.0     -> colors[2]
# ...
# t = 5.0     -> colors[10]
#
# ============================================================

colors = plt.get_cmap(
    "viridis"
)(
    np.linspace(
        0.0,
        1.0,
        len(plot_times) + 1
    )
)


# ============================================================
# Locate PINN columns corresponding to requested times
# ============================================================

selected = []

for t in plot_times:

    index = np.argmin(
        np.abs(
            csv_times - t
        )
    )

    selected.append(
        (
            t,
            index,
            time_columns[index]
        )
    )


# ============================================================
# Calculate error histories
# ============================================================

linf_error = []
relative_l2_error = []


for i, t in enumerate(csv_times):

    # --------------------------------------------------------
    # PINN solution
    # --------------------------------------------------------

    u_pinn = U_pinn[:, i]


    # --------------------------------------------------------
    # Exact solution
    # --------------------------------------------------------

    u_ex = u_exact(
        x,
        t
    )


    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    error = (
        u_pinn
        -
        u_ex
    )


    # --------------------------------------------------------
    # Maximum absolute error
    # --------------------------------------------------------

    linf = np.max(
        np.abs(error)
    )

    linf_error.append(
        linf
    )


    # --------------------------------------------------------
    # Discrete Relative L2 error
    # --------------------------------------------------------

    rel_l2 = (
        np.sqrt(
            np.sum(
                error**2
            )
        )
        /
        np.sqrt(
            np.sum(
                u_ex**2
            )
        )
    )

    relative_l2_error.append(
        rel_l2
    )


# Convert to arrays

linf_error = np.asarray(
    linf_error
)

relative_l2_error = np.asarray(
    relative_l2_error
)


# ============================================================
# FIGURE 1
# Exact + PINN solution profiles
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 7)
)


# ------------------------------------------------------------
# Exact solution at t = 0
# ------------------------------------------------------------

ax.plot(
    x,
    u_exact(
        x,
        0.0
    ),
    color=colors[0],
    linewidth=2.0,
    linestyle="-",
    label=r"Exact, $t=0$"
)


# ------------------------------------------------------------
# Exact + PINN at t = 0.5,...,5.0
# ------------------------------------------------------------

for j, (
    t,
    index,
    column
) in enumerate(selected):

    color = colors[j + 1]


    # --------------------------------------------------------
    # Exact solution
    # --------------------------------------------------------

    u_ex = u_exact(
        x,
        t
    )

    ax.plot(
        x,
        u_ex,
        color=color,
        linewidth=1.5,
        linestyle="-",
        label=fr"Exact, $t={t:g}$"
    )


    # --------------------------------------------------------
    # PINN solution
    # --------------------------------------------------------

    u_pinn = U_pinn[
        :,
        index
    ]

    ax.plot(
        x,
        u_pinn,
        color=color,
        linewidth=1.5,
        linestyle="--",
        label=fr"PINN, $t={t:g}$"
    )


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax.set_xlabel(
    r"$x$",
    fontsize=14
)

plt.ylabel(
    r"$\phi(x,t)$",
    fontsize=14
)


# ------------------------------------------------------------
# Tick formatting
# ------------------------------------------------------------

ax.tick_params(
    axis="both",
    labelsize=12
)


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

ax.grid(
    True,
    alpha=0.3,
    linewidth=0.8
)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

ax.legend(
    fontsize=9,
    ncol=2,
    frameon=True,
    loc="upper right"
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

plt.tight_layout(
    pad=1.5
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
plt.savefig(
    "P4_DT_PINN_solution.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# FIGURE 2
# Absolute error profiles
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 7)
)


for j, (
    t,
    index,
    column
) in enumerate(selected):

    color = colors[j + 1]


    # --------------------------------------------------------
    # PINN solution
    # --------------------------------------------------------

    u_pinn = U_pinn[
        :,
        index
    ]


    # --------------------------------------------------------
    # Exact solution
    # --------------------------------------------------------

    u_ex = u_exact(
        x,
        t
    )


    # --------------------------------------------------------
    # Absolute error
    # --------------------------------------------------------

    error = np.abs(
        u_pinn - u_ex
    )


    ax.plot(
        x,
        error,
        color=color,
        linewidth=1.5,
        label=fr"$t={t:g}$"
    )


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax.set_xlabel(
    r"$x$",
    fontsize=15
)

ax.set_ylabel(
    "Absolute error",
    fontsize=15
)


# ------------------------------------------------------------
# Tick formatting
# ------------------------------------------------------------

ax.tick_params(
    axis="both",
    labelsize=12
)

formatter = ScalarFormatter(
    useMathText=True
)

formatter.set_powerlimits(
    (0, 0)
)

ax.yaxis.set_major_formatter(
    formatter
)


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

ax.grid(
    True,
    alpha=0.3,
    linewidth=0.8
)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

ax.legend(
    fontsize=9,
    ncol=2,
    frameon=True
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

plt.tight_layout(
    pad=1.5
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

plt.savefig(
    "P4_DT_PINN_absolute_error.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()
# ============================================================
# FIGURE 3
# Error versus time
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 7)
)


# ------------------------------------------------------------
# Maximum absolute error
# ------------------------------------------------------------

ax.semilogy(
    csv_times,
    linf_error,
    color="black",
    linewidth=1.5,
    label="Maximum absolute error"
)


# ------------------------------------------------------------
# Relative L2 error
# ------------------------------------------------------------

ax.semilogy(
    csv_times,
    relative_l2_error,
    color="red",
    linewidth=1.5,
    label=r"Relative $L_2$ error"
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax.set_xlabel(
    "Time",
    fontsize=14
)

ax.set_ylabel(
    "Error",
    fontsize=14
)


# ------------------------------------------------------------
# Tick formatting
# ------------------------------------------------------------

ax.tick_params(
    axis="both",
    labelsize=12
)


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

ax.grid(
    True,
    alpha=0.3,
    linewidth=0.8
)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

ax.legend(
    fontsize=11,
    frameon=True
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

plt.tight_layout(
    pad=1.5
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
plt.savefig(
    "P4_DT_PINN_error_vs_time.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Final errors
# ============================================================

print("\n============================================")
print("PINN ERROR SUMMARY")
print("============================================")

print(
    f"Maximum absolute error at t = 5 : "
    f"{linf_error[-1]:.2e}"
)

print(
    f"Relative L2 error at t = 5       : "
    f"{relative_l2_error[-1]:.2e}"
)


# ============================================================
# Overall discrete space-time Relative L2 error
# ============================================================
#
# This is calculated over ALL CSV time levels and
# ALL spatial points:
#
#       sqrt(sum_n sum_i error^2)
#       --------------------------
#       sqrt(sum_n sum_i exact^2)
#
# ============================================================

overall_error_sq = 0.0
overall_exact_sq = 0.0


for i, t in enumerate(csv_times):

    u_pinn = U_pinn[
        :,
        i
    ]

    u_ex = u_exact(
        x,
        t
    )

    error = (
        u_pinn
        -
        u_ex
    )

    overall_error_sq += np.sum(
        error**2
    )

    overall_exact_sq += np.sum(
        u_ex**2
    )


overall_rel_l2 = (
    np.sqrt(
        overall_error_sq
    )
    /
    np.sqrt(
        overall_exact_sq
    )
)


print(
    f"Overall discrete Relative L2 : "
    f"{overall_rel_l2:.2e}"
)

print("============================================")


# In[ ]:




