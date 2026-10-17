#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Plot settings
# ==========================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


# ==========================================================
# CSV files
# ==========================================================

# Files:
# data/t_0000.csv
# data/t_0001.csv
# ...
# data/t_0500.csv

files = [
    f"data/t_{i:04d}.csv"
    for i in range(501)
]


if len(files) != 501:

    raise ValueError(
        f"Expected 501 CSV files, "
        f"but found {len(files)}."
    )


# ==========================================================
# Storage
# ==========================================================

times = []

exact_profiles = []

pinn_profiles = []

fem_profiles = []

pinn_error_profiles = []

fem_error_profiles = []

x_grid = None


# ==========================================================
# Load all 501 CSV files
# ==========================================================

for index in range(501):

    # ------------------------------------------------------
    # File name
    # ------------------------------------------------------

    file = f"data/t_{index:04d}.csv"


    # ------------------------------------------------------
    # Extract time
    #
    # t_0000.csv -> 0.00
    # t_0001.csv -> 0.01
    # ...
    # t_0500.csv -> 5.00
    # ------------------------------------------------------

    t = index * 0.01


    # ------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------

    df = pd.read_csv(file)


    # ------------------------------------------------------
    # Check columns
    # ------------------------------------------------------

    required = [
        "x",
        "Exact",
        "PINN",
        "FEM"
    ]

    for col in required:

        if col not in df.columns:

            raise ValueError(
                f"{file} does not contain "
                f"column '{col}'."
            )


    # ------------------------------------------------------
    # Sort by x
    # ------------------------------------------------------

    df = df.sort_values(
        "x"
    )


    # ------------------------------------------------------
    # Average duplicate x values
    # ------------------------------------------------------

    grouped = (
        df.groupby(
            "x",
            as_index=False
        )[["FEM", "Exact", "PINN"]]
        .mean()
    )

    x = grouped["x"].values

    fem = grouped["FEM"].values

    exact = grouped["Exact"].values

    pinn = grouped["PINN"].values


    # ------------------------------------------------------
    # Common x-grid
    # ------------------------------------------------------

    if x_grid is None:

        x_grid = x

    else:

        if not np.array_equal(
            x_grid,
            x
        ):

            raise ValueError(
                f"x-grid mismatch detected in {file}."
            )


    # ------------------------------------------------------
    # Absolute errors
    # ------------------------------------------------------

    pinn_error = np.abs(
        exact - pinn
    )

    fem_error = np.abs(
        exact - fem
    )


    # ------------------------------------------------------
    # Store
    # ------------------------------------------------------

    times.append(t)

    exact_profiles.append(
        exact
    )

    pinn_profiles.append(
        pinn
    )

    fem_profiles.append(
        fem
    )

    pinn_error_profiles.append(
        pinn_error
    )

    fem_error_profiles.append(
        fem_error
    )


# ==========================================================
# Convert to arrays
# ==========================================================

times = np.asarray(times)

exact_profiles = np.asarray(
    exact_profiles
)

pinn_profiles = np.asarray(
    pinn_profiles
)

fem_profiles = np.asarray(
    fem_profiles
)

pinn_error_profiles = np.asarray(
    pinn_error_profiles
)

fem_error_profiles = np.asarray(
    fem_error_profiles
)


# ==========================================================
# Common x-grid
# ==========================================================

x = x_grid


# ==========================================================
# Space-time mesh
# ==========================================================

X, TT = np.meshgrid(
    x,
    times
)


# ==========================================================
# GLOBAL ERRORS
# ==========================================================

# ----------------------------------------------------------
# PINN: include t = 0
# ----------------------------------------------------------

PINN_RMS = np.sqrt(
    np.mean(
        pinn_error_profiles ** 2
    )
)

PINN_Linf = np.max(
    pinn_error_profiles
)

PINN_RelL2 = (
    np.linalg.norm(
        pinn_error_profiles
    )
    /
    np.linalg.norm(
        exact_profiles
    )
)


# ----------------------------------------------------------
# FEM: exclude t = 0
# ----------------------------------------------------------

fem_error_profiles_no_t0 = fem_error_profiles[1:]
exact_profiles_no_t0 = exact_profiles[1:]

FEM_RMS = np.sqrt(
    np.mean(
        fem_error_profiles_no_t0 ** 2
    )
)

FEM_Linf = np.max(
    fem_error_profiles_no_t0
)

FEM_RelL2 = (
    np.linalg.norm(
        fem_error_profiles_no_t0
    )
    /
    np.linalg.norm(
        exact_profiles_no_t0
    )
)

print()
print("==========================================================")
print("                 GLOBAL ERROR SUMMARY")
print("==========================================================")

print()
print("PINN:")
print(
    f"RMS error         = {PINN_RMS:.2e}"
)
print(
    f"Linf error        = {PINN_Linf:.2e}"
)
print(
    f"Relative L2 error = {PINN_RelL2:.2e}"
)

print()
print("FEM:")
print(
    f"RMS error         = {FEM_RMS:.2e}"
)
print(
    f"Linf error        = {FEM_Linf:.2e}"
)
print(
    f"Relative L2 error = {FEM_RelL2:.2e}"
)

print()
print("==========================================================")
print("                 DATA INFORMATION")
print("==========================================================")

print(
    f"Number of time levels = {len(times)}"
)

print(
    f"Number of spatial points = {len(x)}"
)

print(
    f"Time range = "
    f"{times.min():.2f} to {times.max():.2f}"
)

print("==========================================================")


# ==========================================================
# ==========================================================
# 1. PINN SOLUTION CONTOUR
# ==========================================================
# ==========================================================

plt.figure(
    figsize=(7, 5)
)

cp = plt.contourf(
    X,
    TT,
    pinn_profiles,
    100,
    cmap="jet"
)

plt.colorbar(cp)

plt.xlabel("x")

plt.ylabel("t")

# plt.title(
#     "PINN",
#     fontsize=15
# )

plt.tight_layout()

plt.savefig(
    "P4_CT_PINN_cont.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# ==========================================================
# 2. PINN ABSOLUTE ERROR CONTOUR
# ==========================================================
# ==========================================================

plt.figure(
    figsize=(7, 5)
)

cp = plt.contourf(
    X,
    TT,
    pinn_error_profiles,
    100,
    cmap="jet"
)

plt.colorbar(cp)

plt.xlabel("x")

plt.ylabel("t")

# plt.title(
#     "PINN Absolute Error",
#     fontsize=15
# )

plt.tight_layout()

# plt.savefig(
#     "PINN_error_contour.pdf",
#     dpi=400,
#     bbox_inches="tight"
# )

plt.show()


# ==========================================================
# ==========================================================
# 3. PINN SOLUTION SURFACE
# ==========================================================
# ==========================================================

fig = plt.figure(
    figsize=(8, 7)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

surf = ax.plot_surface(
    X,
    TT,
    pinn_profiles,
    cmap="jet",
    linewidth=0
)

ax.set_xlabel("x")

ax.set_ylabel("t")

# ax.set_title(
#     "PINN"
# )

cbar = fig.colorbar(
    surf,
    ax=ax,
    shrink=0.7
)

cbar.set_label(
    "PINN solution"
)

plt.tight_layout()

# plt.savefig(
#     "PINN_solution_surface.pdf",
#     dpi=400,
#     bbox_inches="tight"
# )

plt.show()


# ==========================================================
# ==========================================================
# 4. PINN ABSOLUTE ERROR SURFACE
# ==========================================================
# ==========================================================

fig = plt.figure(
    figsize=(8, 7)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

surf = ax.plot_surface(
    X,
    TT,
    pinn_error_profiles,
    cmap="jet",
    linewidth=0,
    vmin=np.min(pinn_error_profiles),
    vmax=np.max(pinn_error_profiles)
)

ax.set_xlabel("x")

ax.set_ylabel("t")

cbar = fig.colorbar(
    surf,
    ax=ax,
    shrink=0.7
)

cbar.set_label(
    "Absolute error"
)

ax.set_zticklabels([])

plt.tight_layout()

plt.savefig(
    "P4_CT_PINN_abs_error.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# SOLUTION PROFILES
# t = 0, 0.5, 1.0, ..., 5.0
# ==========================================================

plot_times = np.arange(
    0.0,
    5.01,
    0.5
)

cmap = plt.get_cmap("viridis")

colors = cmap(
    np.linspace(
        0,
        1,
        len(plot_times)
    )
)

plt.figure(
    figsize=(9, 7)
)

for t_val, color in zip(
    plot_times,
    colors
):

    # ------------------------------------------------------
    # Find nearest available time level
    # ------------------------------------------------------

    row = np.argmin(
        np.abs(times - t_val)
    )

    actual_time = times[row]

    # ------------------------------------------------------
    # Exact
    # ------------------------------------------------------

    plt.plot(
        x,
        exact_profiles[row],
        color=color,
        linewidth=1.5,
        linestyle="-",
        label=fr"Exact, $t={actual_time:g}$"
    )

    # ------------------------------------------------------
    # PINN
    # ------------------------------------------------------

    plt.plot(
        x,
        pinn_profiles[row],
        color=color,
        linewidth=1.5,
        linestyle="--",
        label=fr"PINN, $t={actual_time:g}$"
    )


plt.xlabel("x")

plt.ylabel(
    r"$\phi(x,t)$"
)

plt.grid(True)

plt.legend(
    ncol=2,
    fontsize=10
)

plt.tight_layout()

plt.savefig(
    "P4_CT_PINN_sol_profile.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()


# In[ ]:




