# ---------------------------------------------------------------
# FIXED CHARGE TRANSPORTATION PROBLEM (FCTP)
# DATA GENERATION FOR MULTIPLE (B, θ) COMBINATIONS
# Each combination stored as a separate sheet in ONE Excel file
# ---------------------------------------------------------------

import numpy as np
import pandas as pd

# -------------------------------
# 1. GLOBAL PARAMETERS
# -------------------------------

m = 40                   # number of sources
n = 40                   # number of sinks
B_values = [10]  # supply/demand upper bounds
theta_values = [0.60]  # flow ratio parameters
f_lb = 200              # lower bound of fixed cost
seed = 42              # random seed for reproducibility

# -------------------------------
# 2. OUTPUT EXCEL SETUP
# -------------------------------

output_filename = "Input_data_40_40.xlsx"
writer = pd.ExcelWriter(output_filename, engine="openpyxl")

# -------------------------------
# 3. LOOP THROUGH (B, θ) COMBINATIONS
# -------------------------------

for B in B_values:
    for theta in theta_values:

        # --- Supplies and Demands ---
        a_i = np.random.randint(1, B + 1, size=m)
        b_j = np.random.randint(1, B + 1, size=n)

        # Balance total supply & demand
        D = a_i.sum()
        scale = D / b_j.sum()
        b_j = np.round(b_j * scale).astype(int)
        residual = D - b_j.sum()
        if residual != 0:
            b_j[-1] += residual
        assert a_i.sum() == b_j.sum(), "Supply and demand not balanced!"

        # --- Fixed & Variable Costs ---
        f_ij = np.random.randint(f_lb, 801, size=(m, n))
        c_ij_random = np.random.randint(200, 801, size=(m, n))
        c_ij_scaled = np.round(theta * f_ij * (m + n - 1) / D, 2)

        # --- Profit per Sink ---
        profit_j = np.random.randint(800, 1501, size=n)

        # --- Combine into single DataFrame ---
        data = []
        for i in range(m):
            for j in range(n):
                data.append({
                    "Source": f"S{i+1}",
                    "Sink": f"D{j+1}",
                    "FixedCost": f_ij[i, j],
                    "VariableCost_Random": c_ij_random[i, j],
                    "VariableCost": c_ij_scaled[i, j],
                    "a_i": a_i[i],
                    "b_j": b_j[j],
                    "profit_j": profit_j[j],
                    "B": B,
                    "θ": theta
                })

        df = pd.DataFrame(data) 

        # --- Sheet name ---
        sheet_name = f"B{B}_θ{theta}".replace('.', 'p')
        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

# -------------------------------
# 4. SAVE FINAL FILE
# -------------------------------
writer.close()

# -------------------------------
# 5. DISPLAY SUMMARY
# -------------------------------
print("\n✅ All FCTP data instances generated successfully.")
print(f"📁 Saved as '{output_filename}' with {len(B_values) * len(theta_values)} sheets.")
print("\n=== Sheet Names ===")
for B in B_values:
    for theta in theta_values:
        print(f" - B{B}_θ{theta}")