#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Problem parameters
# ==========================================================

Lx = 2.0
Ly = 2.0

k = 0.01

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
# Load data
# ==========================================================

df = pd.read_csv("data.csv")

print("Data shape:", df.shape)
print("Number of spatial points:", len(df))


# ==========================================================
# Check coordinate columns
# ==========================================================

if "x" not in df.columns or "y" not in df.columns:
    raise ValueError("CSV must contain 'x' and 'y' columns.")


# ==========================================================
# Identify DT-PINN time columns
#
# dt = 0.001 s
# Results saved every 10 computational time steps
#
# Therefore:
#
# t0001 -> 0.01 s
# t0002 -> 0.02 s
# ...
# t0100 -> 1.00 s
#
# t0000 is NOT used.
# ==========================================================

time_columns = []

for col in df.columns:

    if col.startswith("t") and col[1:].isdigit():

        index = int(col[1:])

        if 1 <= index <= 100:
            time_columns.append(col)


# Sort chronologically
time_columns = sorted(
    time_columns,
    key=lambda c: int(c[1:])
)


# ==========================================================
# Check available time columns
# ==========================================================

print("\nNumber of DT-PINN time snapshots:", len(time_columns))

if len(time_columns) != 100:

    print(
        "WARNING: Expected 100 time columns "
        "(t0001 to t0100)."
    )

print(
    "First snapshot:",
    time_columns[0],
    "->",
    int(time_columns[0][1:]) * 0.01,
    "s"
)

print(
    "Last snapshot:",
    time_columns[-1],
    "->",
    int(time_columns[-1][1:]) * 0.01,
    "s"
)


# ==========================================================
# Coordinates
# ==========================================================

x = df["x"].values
y = df["y"].values


# ==========================================================
# Calculate errors at every stored time step
# ==========================================================

times = []

max_abs_errors = []

relative_l2_errors = []


for column in time_columns:

    # ------------------------------------------------------
    # Physical time
    # ------------------------------------------------------

    output_index = int(column[1:])

    t = output_index * 0.01


    # ------------------------------------------------------
    # DT-PINN solution
    # ------------------------------------------------------

    U_num = df[column].values


    # ------------------------------------------------------
    # Exact solution
    # ------------------------------------------------------

    U_ex = u_exact(
        x,
        y,
        t
    )


    # ------------------------------------------------------
    # Error
    # ------------------------------------------------------

    error = U_ex - U_num

    abs_error = np.abs(error)


    # ------------------------------------------------------
    # Maximum absolute error
    # ------------------------------------------------------

    max_abs_error = np.max(
        abs_error
    )


    # ------------------------------------------------------
    # Relative L2 error
    #
    # ||U_exact - U_num||_2
    # ----------------------
    #       ||U_exact||_2
    # ------------------------------------------------------

    relative_l2_error = (
        np.linalg.norm(error)
        /
        np.linalg.norm(U_ex)
    )


    # ------------------------------------------------------
    # Store
    # ------------------------------------------------------

    times.append(t)

    max_abs_errors.append(
        max_abs_error
    )

    relative_l2_errors.append(
        relative_l2_error
    )


# ==========================================================
# Convert to NumPy arrays
# ==========================================================

times = np.array(times)

max_abs_errors = np.array(
    max_abs_errors
)

relative_l2_errors = np.array(
    relative_l2_errors
)


# ==========================================================
# Print selected time levels
# ==========================================================

print("\nSelected error values:")

for target_time in [0.2, 0.4, 0.6, 0.8, 1.0]:

    idx = np.argmin(
        np.abs(times - target_time)
    )

    print(
        f"t = {times[idx]:.2f} s   "
        f"Linf = {max_abs_errors[idx]:.2e}   "
        f"Relative L2 = {relative_l2_errors[idx]:.2e}"
    )


# ==========================================================
# Overall Relative L2 Error
#
# Uses ALL spatial points and ALL 100 stored time levels.
#
#                 sqrt(sum_n ||u_exact^n-u_num^n||_2^2)
# E_overall = --------------------------------------------
#                 sqrt(sum_n ||u_exact^n||_2^2)
# ==========================================================

sum_squared_error = 0.0
sum_squared_exact = 0.0


for column in time_columns:

    output_index = int(column[1:])

    t = output_index * 0.01

    U_num = df[column].values

    U_ex = u_exact(
        x,
        y,
        t
    )

    sum_squared_error += np.sum(
        (U_ex - U_num) ** 2
    )

    sum_squared_exact += np.sum(
        U_ex ** 2
    )


overall_relative_l2 = (
    np.sqrt(sum_squared_error)
    /
    np.sqrt(sum_squared_exact)
)


print(
    "\n=================================================="
)

print(
    f"Overall Relative L2 Error "
    f"(0.01 s to 1.00 s) = "
    f"{overall_relative_l2:.2e}"
)

print(
    "=================================================="
)


# ==========================================================
# Plot maximum absolute error and relative L2 error
# on ONE plot
# ==========================================================

fig = plt.figure(
    figsize=(9, 7)
)


plt.semilogy(
    times,
    max_abs_errors,
    color="black",
    linewidth=1.5,
    label="Maximum absolute error"
)

plt.semilogy(
    times,
    relative_l2_errors,
    color="red",
    linewidth=1.5,
    label=r"Relative $L_2$ error"
)


# ==========================================================
# Labels
# ==========================================================

plt.xlabel(
    "Time",
    fontsize=14
)

plt.ylabel(
    "Error",
    fontsize=14
)


# ==========================================================
# X-axis
# ==========================================================

plt.xticks(
    np.arange(0.0, 1.01, 0.1),
    fontsize=11
)


plt.yticks(
    fontsize=11
)


# ==========================================================
# Grid
# ==========================================================

plt.grid(
    True,
    which="both",
    alpha=0.3
)


# ==========================================================
# Legend
# ==========================================================

plt.legend(
    fontsize=11
)


plt.tight_layout()


# ==========================================================
# Save plot only
# ==========================================================

plt.savefig(
    "P5_DT_PINN_error_vs_time.pdf",
    dpi=400,
    bbox_inches="tight"
)


plt.close(fig)


print(
    "\nPlot saved as:"
    "\nP5_DT_PINN_error_vs_time.pdf"
)


# In[ ]:




