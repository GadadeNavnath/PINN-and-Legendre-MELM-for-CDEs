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

melm = pd.read_csv("P3_MELM.csv")
ref = pd.read_csv("P3_PINN.csv", sep=";")

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
print(f"Matched points : {mask.sum()}")
print(f"Unmatched      : {(~mask).sum()}")
print(f"Maximum distance : {dist[mask].max():.3e}")

# ==========================================================
# Construct matched dataframe
# ==========================================================

matched = pd.DataFrame({
    "X": melm.loc[mask, "X"].to_numpy(),
    "Y": melm.loc[mask, "Y"].to_numpy(),
    "FEM": ref.iloc[idx[mask]]["FEM"].to_numpy(),
    "PINN": ref.iloc[idx[mask]]["PINN"].to_numpy(),
    "MELM": melm.loc[mask, "U"].to_numpy()
})

matched["PINN_Abs_Error"] = np.abs(
    matched["PINN"] - matched["FEM"]
)

matched["MELM_Abs_Error"] = np.abs(
    matched["MELM"] - matched["FEM"]
)

# ==========================================================
# Convert to structured grid
# ==========================================================

x = np.sort(matched["X"].unique())
y = np.sort(matched["Y"].unique())

X, Y = np.meshgrid(x, y)

FEM = matched.pivot(index="Y", columns="X",
                    values="FEM").loc[y, x].values

PINN = matched.pivot(index="Y", columns="X",
                     values="PINN").loc[y, x].values

MELM = matched.pivot(index="Y", columns="X",
                     values="MELM").loc[y, x].values

PINN_ERR = matched.pivot(index="Y", columns="X",
                         values="PINN_Abs_Error").loc[y, x].values

MELM_ERR = matched.pivot(index="Y", columns="X",
                         values="MELM_Abs_Error").loc[y, x].values

# ==========================================================
# Common limits
# ==========================================================

vmin = min(FEM.min(), PINN.min(), MELM.min())
vmax = max(FEM.max(), PINN.max(), MELM.max())

levels = np.linspace(vmin, vmax, 60)

err_vmin = 0.0
err_vmax = max(PINN_ERR.max(), MELM_ERR.max())

# ==========================================================
# 1. Surface plots
# ==========================================================

fig = plt.figure(
    figsize=(18,6),
    constrained_layout=True
)

titles = [
    "FEM",
    "PINN",
    "MELM"
]

solutions = [
    FEM,
    PINN,
    MELM
]

for i, (Z, title) in enumerate(zip(solutions, titles), start=1):

    ax = fig.add_subplot(1,3,i,projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title, fontsize=15)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")

    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

# -------- Shared colorbar --------
cbar = fig.colorbar(
    surf,
    ax=fig.axes,
    shrink=0.72,
    pad=0.03
)

cbar.set_label(r"$\phi$", fontsize=13)

plt.savefig("P3_surface.pdf", dpi=600, bbox_inches="tight")

# ==========================================================
# 2. Contour plots
# ==========================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18,5),
    constrained_layout=True
)

for ax, Z, title in zip(axes, solutions, titles):

    cf = ax.contourf(
        X,
        Y,
        Z,
        levels=levels,
        cmap="jet",
        extend="both"
    )

    ax.set_title(title, fontsize=15)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_aspect("equal")

    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

# -------- Shared colorbar --------
cbar = fig.colorbar(
    cf,
    ax=axes,
    shrink=1,
    pad=0.02
)

cbar.set_label(r"$\phi$", fontsize=13)

plt.savefig("P3_contour.pdf", dpi=600, bbox_inches="tight")

# ==========================================================
# 3. Absolute error surfaces
# ==========================================================

fig = plt.figure(figsize=(14,6))

error_titles = [
    "PINN",
    "MELM"
]

errors = [
    PINN_ERR,
    MELM_ERR
]

for i, (E, title) in enumerate(zip(errors, error_titles), start=1):

    ax = fig.add_subplot(1,2,i,projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        E,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        vmin=err_vmin,
        vmax=err_vmax
    )

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Absolute error")

    ax.xaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )

    # ------------------------------------------------------
    # Highlight TWO largest peaks
    # ------------------------------------------------------

    dy = 0.040
    zmax = ax.get_zlim()[1]

    inds = np.argpartition(E.ravel(), -2)[-2:]
    inds = inds[np.argsort(E.ravel()[inds])[::-1]]

    for k in inds:

        ii, jj = np.unravel_index(k, E.shape)

        x0 = X[ii, jj]
        y0 = Y[ii, jj]
        z0 = E[ii, jj]

        # Left edge
        ax.plot(
            [x0, x0],
            [y0-dy, y0-dy],
            [0, zmax],
            color="black",
            linewidth=1
        )

        # Right edge
        ax.plot(
            [x0, x0],
            [y0+dy, y0+dy],
            [0, zmax],
            color="black",
            linewidth=1
        )

        # Bottom edge
        ax.plot(
            [x0, x0],
            [y0-dy, y0+dy],
            [0, 0],
            color="black",
            linewidth=1
        )

        # Top edge
        ax.plot(
            [x0, x0],
            [y0-dy, y0+dy],
            [zmax, zmax],
            color="black",
            linewidth=1
        )

        ax.text(
            x0,
            y0,
            zmax*1.03,
            f"{z0:.2e}",
            color="black",
            fontsize=9,
            ha="center",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                linewidth=1.3,
                pad=0.5
            )
        )

    fig.colorbar(
        surf,
        ax=ax,
        shrink=0.72
    )

plt.tight_layout()

plt.savefig("P3_error.pdf", dpi=600, bbox_inches="tight")

plt.show()

# ==========================================================
# 4. Comparison along x = -0.25, 0, 0.25
# ==========================================================

target_x = [-0.25, 0.0, 0.25]

fig, axes = plt.subplots(1, 3, figsize=(16,5), sharey=True)

for ax, x_target in zip(axes, target_x):

    # Find nearest x-location
    idx = np.argmin(np.abs(x - x_target))
    x_actual = x[idx]

    # FEM
    ax.plot(
        y,
        FEM[:, idx],
        'k-',
        linewidth=1.5,
        label='FEM'
    )

    # PINN
    ax.plot(
        y,
        PINN[:, idx],
        'r--',
        linewidth=1.5,
        label='PINN'
    )

    # MELM
    ax.plot(
        y,
        MELM[:, idx],
        'b-.',
        linewidth=1.5,
        label='MELM'
    )

    ax.set_title(rf"$x={x_actual:.2f}$", fontsize=15)
    ax.set_xlabel("y", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.3)

axes[0].set_ylabel("Solution", fontsize=13)

# ----------------------------------------------------------
# Common legend (only once)
# ----------------------------------------------------------

for ax in axes:
    ax.legend(
        loc='best',          # or 'upper right'
        fontsize=11,
        frameon=True,
        edgecolor='black'
    )

# Improve spacing
plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.savefig("P3_int_com.pdf", dpi=600, bbox_inches="tight")

plt.show()

# ==========================================================
# Boundary comparison
# ==========================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=False)

# ----------------------------------------------------------
# Left boundary (x = -1)
# ----------------------------------------------------------

idx = 0

axes[0,0].plot(y, FEM[:,idx], 'k-', lw=1, label='FEM')
axes[0,0].plot(y, PINN[:,idx], 'r--', lw=1, label='PINN')
axes[0,0].plot(y, MELM[:,idx], 'b-.', lw=1, label='MELM')

axes[0,0].set_title(rf'$x={x[0]:.1f}$', fontsize=14)
axes[0,0].set_xlabel('y', fontsize=12)
axes[0,0].set_ylabel(r'$\phi$', fontsize=13)
axes[0,0].grid(alpha=0.3)

# ----------------------------------------------------------
# Right boundary (x = 1)
# ----------------------------------------------------------

idx = -1

axes[0,1].plot(y, FEM[:,idx], 'k-', lw=1, label='FEM')
axes[0,1].plot(y, PINN[:,idx], 'r--', lw=1, label='PINN')
axes[0,1].plot(y, MELM[:,idx], 'b-.', lw=1, label='MELM')

axes[0,1].set_title(rf'$x={x[-1]:.1f}$', fontsize=14)
axes[0,1].set_xlabel('y', fontsize=12)
axes[0,1].grid(alpha=0.3)

# ----------------------------------------------------------
# Bottom boundary (y = -1)
# ----------------------------------------------------------

idx = 0

axes[1,0].plot(x, FEM[idx,:], 'k-', lw=1, label='FEM')
axes[1,0].plot(x, PINN[idx,:], 'r--', lw=1, label='PINN')
axes[1,0].plot(x, MELM[idx,:], 'b-.', lw=1, label='MELM')

axes[1,0].set_title(rf'$y={y[0]:.1f}$', fontsize=14)
axes[1,0].set_xlabel('x', fontsize=12)
axes[1,0].set_ylabel(r'$\phi$', fontsize=13)
axes[1,0].grid(alpha=0.3)

# ----------------------------------------------------------
# Top boundary (y = 1)
# ----------------------------------------------------------

idx = -1

axes[1,1].plot(x, FEM[idx,:], 'k-', lw=1, label='FEM')
axes[1,1].plot(x, PINN[idx,:], 'r--', lw=1, label='PINN')
axes[1,1].plot(x, MELM[idx,:], 'b-.', lw=1, label='MELM')

axes[1,1].set_title(rf'$y={y[-1]:.1f}$', fontsize=14)
axes[1,1].set_xlabel('x', fontsize=12)
axes[1,1].grid(alpha=0.3)

# ----------------------------------------------------------
# Legend for each subplot
# ----------------------------------------------------------

for ax in axes.flat:
    ax.legend(
        loc='best',
        fontsize=10,
        frameon=True,
        edgecolor='black'
    )
    ax.tick_params(axis='both', labelsize=11)

plt.tight_layout()

plt.savefig("P3_bnd_com.pdf", dpi=600, bbox_inches="tight")

plt.show()

# ==========================================================
# Relative L2 error
# ==========================================================

pinn_rel_l2 = np.linalg.norm(
    matched["PINN"] - matched["FEM"]
) / np.linalg.norm(matched["FEM"])

melm_rel_l2 = np.linalg.norm(
    matched["MELM"] - matched["FEM"]
) / np.linalg.norm(matched["FEM"])

print("=" * 60)
print("Relative L2 Errors")
print("=" * 60)
print(f"PINN Relative L2 Error : {pinn_rel_l2:.2e}")
print(f"MELM Relative L2 Error : {melm_rel_l2:.2e}")
print("=" * 60)

