#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.ticker import FormatStrFormatter


# ==========================================================
# Read data
# ==========================================================

melm = pd.read_csv("P2_MELM.csv")
ref = pd.read_csv("P2_PINN.csv", sep=";")


# ==========================================================
# Match coordinates
# ==========================================================

tree = cKDTree(ref[["X", "Y"]].to_numpy())

dist, idx = tree.query(
    melm[["X", "Y"]].to_numpy(),
    distance_upper_bound=1e-12
)

mask = np.isfinite(dist)

print("=" * 60)
print("Matching Statistics")
print("=" * 60)
print(f"Matched points   : {mask.sum()}")
print(f"Unmatched        : {(~mask).sum()}")
print(f"Maximum distance : {dist[mask].max():.3e}")


# ==========================================================
# Matched data
# ==========================================================

matched = pd.DataFrame({
    "X": melm.loc[mask, "X"].to_numpy(),
    "Y": melm.loc[mask, "Y"].to_numpy(),
    "FEM": ref.iloc[idx[mask]]["FEM"].to_numpy(),
    "PINN": ref.iloc[idx[mask]]["PINN"].to_numpy(),
    "MELM": melm.loc[mask, "U"].to_numpy()
})

matched["PINN_ERR"] = np.abs(matched["PINN"] - matched["FEM"])
matched["MELM_ERR"] = np.abs(matched["MELM"] - matched["FEM"])


# ==========================================================
# Structured grid
# ==========================================================

x = np.sort(matched["X"].unique())
y = np.sort(matched["Y"].unique())

X, Y = np.meshgrid(x, y)


def make_grid(column):
    return (
        matched.pivot(index="Y", columns="X", values=column)
        .loc[y, x]
        .values
    )


FEM = make_grid("FEM")
PINN = make_grid("PINN")
MELM = make_grid("MELM")

PINN_ERR = make_grid("PINN_ERR")
MELM_ERR = make_grid("MELM_ERR")


# ==========================================================
# Common plotting data
# ==========================================================

solutions = [FEM, PINN, MELM]
solution_names = ["FEM", "PINN", "MELM"]

vmin = min(Z.min() for Z in solutions)
vmax = max(Z.max() for Z in solutions)

levels = np.linspace(vmin, vmax, 60)

errors = [PINN_ERR, MELM_ERR]
error_names = ["PINN", "MELM"]

err_vmax = max(E.max() for E in errors)


# ==========================================================
# Common axis formatting
# ==========================================================

def format_xy(ax):
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )


# ==========================================================
# 1. Surface plots
# ==========================================================

fig = plt.figure(figsize=(18, 6))

for i, (Z, name) in enumerate(
    zip(solutions, solution_names), start=1
):

    ax = fig.add_subplot(1, 3, i, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(name, fontsize=15)
    format_xy(ax)
    ax.set_zlabel(r"$\phi$", labelpad=3)

cbar = fig.colorbar(
    surf,
    ax=fig.axes,
    shrink=0.60,
    pad=0.03
)

cbar.set_label(r"$\phi$", fontsize=13)

plt.savefig(
    "P2_surface.pdf",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 2. Contour plots
# ==========================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 6),
    constrained_layout=True
)

for ax, Z, name in zip(
    axes, solutions, solution_names
):

    cf = ax.contourf(
        X, Y, Z,
        levels=levels,
        cmap="jet",
        extend="both"
    )

    ax.set_title(name, fontsize=15)
    format_xy(ax)
    ax.set_aspect("equal")

cbar = fig.colorbar(
    cf,
    ax=axes,
    shrink=0.82,
    pad=0.02
)

cbar.set_label(r"$\phi$", fontsize=13)

plt.savefig(
    "P2_contour.pdf",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 3. Absolute error surfaces
# ==========================================================

fig = plt.figure(figsize=(14, 6))

for i, (E, name) in enumerate(
    zip(errors, error_names), start=1
):

    ax = fig.add_subplot(
        1, 2, i,
        projection="3d"
    )

    surf = ax.plot_surface(
        X, Y, E,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        vmin=0,
        vmax=err_vmax
    )

    ax.set_title(name, fontsize=15)
    format_xy(ax)
    ax.set_zlabel("Absolute error")

    fig.colorbar(
        surf,
        ax=ax,
        shrink=0.72
    )

plt.tight_layout()

plt.savefig(
    "P2_error.pdf",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 4. Comparison along x = -0.25, 0, 0.25
# ==========================================================

target_x = [-0.25, 0.0, 0.25]

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 5),
    sharey=True
)

for ax, x_target in zip(axes, target_x):

    ix = np.argmin(np.abs(x - x_target))
    x_actual = x[ix]

    ax.plot(
        y, FEM[:, ix],
        "k-", lw=1.5,
        label="FEM"
    )

    ax.plot(
        y, PINN[:, ix],
        "r--", lw=1.5,
        label="PINN"
    )

    ax.plot(
        y, MELM[:, ix],
        "b-.", lw=1.5,
        label="MELM"
    )

    ax.set_title(
        rf"$x={x_actual:.2f}$",
        fontsize=15
    )

    ax.set_xlabel("y", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.3)

axes[0].set_ylabel(
    r"$\phi$",
    fontsize=13
)

for ax in axes:
    ax.legend(
        loc="best",
        fontsize=11,
        frameon=True,
        edgecolor="black"
    )

plt.tight_layout()

plt.savefig(
    "P2_int_com.pdf",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 5. Boundary comparison
# ==========================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(10, 8)
)

boundary_data = [
    (y, FEM[:, 0],  PINN[:, 0],  MELM[:, 0],
     rf"$x={x[0]:.1f}$", "y", True),

    (y, FEM[:, -1], PINN[:, -1], MELM[:, -1],
     rf"$x={x[-1]:.1f}$", "y", False),

    (x, FEM[0, :],  PINN[0, :],  MELM[0, :],
     rf"$y={y[0]:.1f}$", "x", True),

    (x, FEM[-1, :], PINN[-1, :], MELM[-1, :],
     rf"$y={y[-1]:.1f}$", "x", False)
]


for ax, (coord, fem, pinn, melm_data,
          title, xlabel, ylabel) in zip(
              axes.flat, boundary_data
          ):

    ax.plot(
        coord, fem,
        "k-", lw=1,
        label="FEM"
    )

    ax.plot(
        coord, pinn,
        "r--", lw=1,
        label="PINN"
    )

    ax.plot(
        coord, melm_data,
        "b-.", lw=1,
        label="MELM"
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)

    if ylabel:
        ax.set_ylabel(
            r"$\phi$",
            fontsize=13
        )

    ax.grid(alpha=0.3)

    ax.legend(
        loc="best",
        fontsize=10,
        frameon=True,
        edgecolor="black"
    )

    ax.tick_params(
        axis="both",
        labelsize=11
    )


plt.tight_layout()

plt.savefig(
    "P2_bnd_com.pdf",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# Relative L2 error
# ==========================================================

pinn_rel_l2 = np.linalg.norm(
    matched["PINN"] - matched["FEM"]
) / np.linalg.norm(
    matched["FEM"]
)

melm_rel_l2 = np.linalg.norm(
    matched["MELM"] - matched["FEM"]
) / np.linalg.norm(
    matched["FEM"]
)

print("=" * 60)
print("Relative L2 Errors")
print("=" * 60)
print(f"PINN Relative L2 Error : {pinn_rel_l2:.2e}")
print(f"MELM Relative L2 Error : {melm_rel_l2:.2e}")
print("=" * 60)


# In[ ]:




