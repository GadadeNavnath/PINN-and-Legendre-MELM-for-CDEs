#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

# ============================================================
# Settings
# ============================================================

data_folder = "data"

N_files = 101

files = [
    os.path.join(
        data_folder,
        f"t_{i:04d}.csv"
    )
    for i in range(N_files)
]

print(f"Total files expected: {len(files)}")


# ============================================================
# Storage
# ============================================================

pinn_all = []
pinn_ref_all = []

fem_all = []
fem_ref_all = []


# ============================================================
# Read all files
# ============================================================

for file in files:

    filename = os.path.basename(file)

    # --------------------------------------------------------
    # Check file exists
    # --------------------------------------------------------

    if not os.path.exists(file):
        raise FileNotFoundError(
            f"File not found: {file}"
        )

    df = pd.read_csv(file)

    pinn = df["PINN"].to_numpy(dtype=float)
    fem = df["FEM"].to_numpy(dtype=float)
    ref = df["Exact"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # PINN: use all 101 files
    # --------------------------------------------------------

    pinn_all.append(pinn)
    pinn_ref_all.append(ref)

    # --------------------------------------------------------
    # FEM: skip t_0000.csv
    # --------------------------------------------------------

    if filename != "t_0000.csv":

        fem_all.append(fem)
        fem_ref_all.append(ref)


# ============================================================
# Combine all space-time points
# ============================================================

pinn_all = np.concatenate(pinn_all)
pinn_ref_all = np.concatenate(pinn_ref_all)

fem_all = np.concatenate(fem_all)
fem_ref_all = np.concatenate(fem_ref_all)


# ============================================================
# Relative L2 error
# ============================================================

pinn_l2 = (
    np.linalg.norm(
        pinn_all - pinn_ref_all
    )
    /
    np.linalg.norm(
        pinn_ref_all
    )
)

fem_l2 = (
    np.linalg.norm(
        fem_all - fem_ref_all
    )
    /
    np.linalg.norm(
        fem_ref_all
    )
)


# ============================================================
# Print information
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL SPACE-TIME RELATIVE L2 ERROR")
print("=" * 70)

print(
    f"PINN files used : {N_files}"
)

print(
    f"FEM files used  : {N_files - 1}"
)

print(
    f"PINN space-time points : {len(pinn_all)}"
)

print(
    f"FEM space-time points  : {len(fem_all)}"
)

print("-" * 70)

print(
    f"PINN Relative L2 Error : {pinn_l2:.2e}"
)

print(
    f"FEM  Relative L2 Error : {fem_l2:.2e}"
)

print("=" * 70)


# In[1]:


import numpy as np
import pandas as pd

# ============================================================
# Settings
# ============================================================

data_folder = "data"

N_files = 101

files = [
    f"{data_folder}/t_{i:04d}.csv"
    for i in range(N_files)
]

print(f"Total files expected: {len(files)}")


# ============================================================
# Storage
# ============================================================

pinn_all = []
pinn_ref_all = []

fem_all = []
fem_ref_all = []


# ============================================================
# Read all files
# ============================================================

for file in files:

    filename = file.split("/")[-1]

    # --------------------------------------------------------
    # Check file exists
    # --------------------------------------------------------

    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File not found: {file}"
        )

    pinn = df["PINN"].to_numpy(dtype=float)
    fem = df["FEM"].to_numpy(dtype=float)
    ref = df["Exact"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # PINN: use all 101 files
    # --------------------------------------------------------

    pinn_all.append(pinn)
    pinn_ref_all.append(ref)

    # --------------------------------------------------------
    # FEM: skip t_0000.csv
    # --------------------------------------------------------

    if filename != "t_0000.csv":

        fem_all.append(fem)
        fem_ref_all.append(ref)


# ============================================================
# Combine all space-time points
# ============================================================

pinn_all = np.concatenate(pinn_all)
pinn_ref_all = np.concatenate(pinn_ref_all)

fem_all = np.concatenate(fem_all)
fem_ref_all = np.concatenate(fem_ref_all)


# ============================================================
# Relative L2 error
# ============================================================

pinn_l2 = (
    np.linalg.norm(
        pinn_all - pinn_ref_all
    )
    /
    np.linalg.norm(
        pinn_ref_all
    )
)

fem_l2 = (
    np.linalg.norm(
        fem_all - fem_ref_all
    )
    /
    np.linalg.norm(
        fem_ref_all
    )
)


# ============================================================
# Print information
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL SPACE-TIME RELATIVE L2 ERROR")
print("=" * 70)

print(
    f"PINN files used : {N_files}"
)

print(
    f"FEM files used  : {N_files - 1}"
)

print(
    f"PINN space-time points : {len(pinn_all)}"
)

print(
    f"FEM space-time points  : {len(fem_all)}"
)

print("-" * 70)

print(
    f"PINN Relative L2 Error : {pinn_l2:.2e}"
)

print(
    f"FEM  Relative L2 Error : {fem_l2:.2e}"
)

print("=" * 70)


# In[ ]:




