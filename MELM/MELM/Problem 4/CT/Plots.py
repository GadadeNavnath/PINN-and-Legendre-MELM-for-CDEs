#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

a = 1.0
k = 0.1

def u_exact(x, t):

    return (
        1.0 / np.sqrt(4.0*t + 1.0)
    ) * np.exp(
        -((x - 1.0 - a*t)**2)
        /
        (k*(4.0*t + 1.0))
    )

# =====================================================
# Load IELM solution
# =====================================================

data = pd.read_csv("P4_CT_IELM.csv")

Nx_test = 201
Nt_test = 101

X = data["X"].values.reshape(Nt_test, Nx_test)
TT = data["T"].values.reshape(Nt_test, Nx_test)
u_pred = data["U"].values.reshape(Nt_test, Nx_test)

u_ex = u_exact(X, TT)

# =====================================================
# Errors
# =====================================================

err = np.abs(u_pred - u_ex)

RMS = np.sqrt(np.mean(err**2))
Linf = np.max(err)
RelL2 = np.linalg.norm(err)/np.linalg.norm(u_ex)

print(f"RMS error         = {RMS:.2e}")
print(f"Linf error        = {Linf:.2e}")
print(f"Relative L2 error = {RelL2:.2e}")

# =====================================================
# Exact solution
# =====================================================

plt.figure(figsize=(7,5))

cp = plt.contourf(
    X,
    TT,
    u_ex,
    100,
    cmap="jet"
)

plt.colorbar(cp)

plt.xlabel("x")
plt.ylabel("t")
# plt.title("Exact", fontsize=15)

plt.tight_layout()
plt.savefig("P4_exact.pdf", bbox_inches="tight")
plt.show()

# =====================================================
# IELM solution
# =====================================================

plt.figure(figsize=(7,5))

cp = plt.contourf(
    X,
    TT,
    u_pred,
    100,
    cmap="jet"
)

plt.colorbar(cp)

plt.xlabel("x")
plt.ylabel("t")
# plt.title("IELM", fontsize=15)

plt.tight_layout()
plt.savefig("P4_CT_IELM_cont.pdf", bbox_inches="tight")
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
    # Find nearest available IELM time level
    # ------------------------------------------------------

    row = np.argmin(
        np.abs(
            TT[:, 0] - t_val
        )
    )

    actual_time = TT[row, 0]

    # ------------------------------------------------------
    # Same spatial points for Exact and IELM
    # ------------------------------------------------------

    x_ielm = X[row, :]

    u_ielm = u_pred[row, :]

    # ------------------------------------------------------
    # Exact solution
    # ------------------------------------------------------

    u_exact_ielm = u_exact(
        x_ielm,
        actual_time
    )

    # ------------------------------------------------------
    # Exact
    # ------------------------------------------------------

    plt.plot(
        x_ielm,
        u_exact_ielm,
        color=color,
        linewidth=1.5,
        linestyle="-",
        label=fr"Exact, $t={actual_time:g}$"
    )

    # ------------------------------------------------------
    # IELM
    # ------------------------------------------------------

    plt.plot(
        x_ielm,
        u_ielm,
        color=color,
        linewidth=1.5,
        linestyle="--",
        label=fr"IELM, $t={actual_time:g}$"
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
    "P4_CT_IELM_sol_profile.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()
# =====================================================
# Absolute error contour
# =====================================================

plt.figure(figsize=(7,5))

cp = plt.contourf(
    X,
    TT,
    err,
    100,
    cmap="jet"
)

plt.colorbar(cp)

plt.xlabel("x")
plt.ylabel("t")
# plt.title("Absolute error")

plt.tight_layout()

# plt.savefig(
#     "fig4_error_contour.pdf",
#     bbox_inches="tight"
# )

plt.show()

# =====================================================
# 3D Error surface
# =====================================================

fig = plt.figure(figsize=(8,7))

ax = fig.add_subplot(
    111,
    projection='3d'
)

surf = ax.plot_surface(
    X,
    TT,
    err,
    cmap='jet',
    linewidth=0,
    vmin=np.min(err),
    vmax=np.max(err)
)

ax.set_xlabel('x')
ax.set_ylabel('t')

# ax.set_title('Absolute error')

cbar = fig.colorbar(
    surf,
    ax=ax,
    shrink=0.7
)

cbar.set_label('Absolute error')

ax.set_zticklabels([])

plt.tight_layout()

plt.savefig(
    "P4_CT_IELM_abs_error.pdf",
    bbox_inches='tight'
)

plt.show()


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

a = 1.0
k = 0.1

def u_exact(x, t):

    return (
        1.0 / np.sqrt(4.0*t + 1.0)
    ) * np.exp(
        -((x - 1.0 - a*t)**2)
        /
        (k*(4.0*t + 1.0))
    )

# =====================================================
# Load MELM solution
# =====================================================

data = pd.read_csv("P4_CT_MELM.csv")

Nx_test = 201
Nt_test = 101

X = data["X"].values.reshape(Nt_test, Nx_test)
TT = data["T"].values.reshape(Nt_test, Nx_test)
u_pred = data["U"].values.reshape(Nt_test, Nx_test)

u_ex = u_exact(X, TT)

# =====================================================
# Errors
# =====================================================

err = np.abs(u_pred - u_ex)

RMS = np.sqrt(np.mean(err**2))
Linf = np.max(err)
RelL2 = np.linalg.norm(err)/np.linalg.norm(u_ex)

print(f"RMS error         = {RMS:.2e}")
print(f"Linf error        = {Linf:.2e}")
print(f"Relative L2 error = {RelL2:.2e}")

# =====================================================
# Exact solution
# =====================================================

plt.figure(figsize=(7,5))

cp = plt.contourf(
    X,
    TT,
    u_ex,
    100,
    cmap="jet"
)

plt.colorbar(cp)

plt.xlabel("x")
plt.ylabel("t")
# plt.title("Exact", fontsize=15)

plt.tight_layout()
plt.savefig("P4_exact.pdf", bbox_inches="tight")
plt.show()

# =====================================================
# MELM solution
# =====================================================

plt.figure(figsize=(7,5))

cp = plt.contourf(
    X,
    TT,
    u_pred,
    100,
    cmap="jet"
)

plt.colorbar(cp)

plt.xlabel("x")
plt.ylabel("t")
# plt.title("MELM", fontsize=15)

plt.tight_layout()
plt.savefig("P4_CT_MELM_cont.pdf", bbox_inches="tight")
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
    # Find nearest available MELM time level
    # ------------------------------------------------------

    row = np.argmin(
        np.abs(
            TT[:, 0] - t_val
        )
    )

    actual_time = TT[row, 0]

    # ------------------------------------------------------
    # Same spatial points for Exact and MELM
    # ------------------------------------------------------

    x_melm = X[row, :]

    u_melm = u_pred[row, :]

    # ------------------------------------------------------
    # Exact solution
    # ------------------------------------------------------

    u_exact_melm = u_exact(
        x_melm,
        actual_time
    )

    # ------------------------------------------------------
    # Exact
    # ------------------------------------------------------

    plt.plot(
        x_melm,
        u_exact_melm,
        color=color,
        linewidth=1.5,
        linestyle="-",
        label=fr"Exact, $t={actual_time:g}$"
    )

    # ------------------------------------------------------
    # MELM
    # ------------------------------------------------------

    plt.plot(
        x_melm,
        u_melm,
        color=color,
        linewidth=1.5,
        linestyle="--",
        label=fr"MELM, $t={actual_time:g}$"
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
    "P4_CT_MELM_sol_profile.pdf",
    dpi=400,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# Absolute error contour
# =====================================================

plt.figure(figsize=(7,5))

cp = plt.contourf(
    X,
    TT,
    err,
    100,
    cmap="jet"
)

plt.colorbar(cp)

plt.xlabel("x")
plt.ylabel("t")
# plt.title("Absolute error")

plt.tight_layout()

# plt.savefig(
#     "fig4_error_contour.pdf",
#     bbox_inches="tight"
# )

plt.show()

# =====================================================
# 3D Error surface
# =====================================================

fig = plt.figure(figsize=(8,7))

ax = fig.add_subplot(
    111,
    projection='3d'
)

surf = ax.plot_surface(
    X,
    TT,
    err,
    cmap='jet',
    linewidth=0,
    vmin=np.min(err),
    vmax=np.max(err)
)

ax.set_xlabel('x')
ax.set_ylabel('t')

# ax.set_title('Absolute error')

cbar = fig.colorbar(
    surf,
    ax=ax,
    shrink=0.7
)

cbar.set_label('Absolute error')

ax.set_zticklabels([])

plt.tight_layout()

plt.savefig(
    "P4_CT_MELM_abs_error.pdf",
    bbox_inches='tight'
)

plt.show()


# In[ ]:




