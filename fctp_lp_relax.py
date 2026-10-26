# fctp_lp_relaxation.py
# ---------------------------------------------------------------
# Solve LP-relaxation of the Fixed Charge Transportation Problem
# y_ij ∈ [0,1], maximize total revenue - total costs
# Input Excel columns:
# Source, Sink, FixedCost, VariableCost, a_i, b_j, profit_j, B, θ
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from datetime import datetime
from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, NonNegativeReals, Reals,
    Constraint, Objective, SolverFactory, value, maximize
)

# ---------------------------------------------------------------
# 1. INPUT FILE
# ---------------------------------------------------------------
file_path = "Input_data_20251103_164929.xlsx"   # <-- Change filename if needed

# ---------------------------------------------------------------
# 2. READ EXCEL DATA
# ---------------------------------------------------------------
df = pd.read_excel(file_path, sheet_name=0)
df.columns = [c.strip() for c in df.columns]

required_cols = ["Source", "Sink", "FixedCost", "VariableCost", "a_i", "b_j", "profit_j", "B", "θ"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in Excel: {missing}")

# preserve order
sources = list(pd.Categorical(df["Source"], categories=df["Source"].unique(), ordered=True).categories)
sinks = list(pd.Categorical(df["Sink"], categories=df["Sink"].unique(), ordered=True).categories)
m, n = len(sources), len(sinks)

# maps
source_map = {s: i for i, s in enumerate(sources)}
sink_map = {t: j for j, t in enumerate(sinks)}

# parameters
a_i_series = df.groupby("Source")["a_i"].first()
b_j_series = df.groupby("Sink")["b_j"].first()

a_i = {source_map[s]: float(a_i_series[s]) for s in sources}
b_j = {sink_map[t]: float(b_j_series[t]) for t in sinks}

f_ij, c_ij, p_j = {}, {}, {}
for _, row in df.iterrows():
    i = source_map[row["Source"]]
    j = sink_map[row["Sink"]]
    f_ij[(i, j)] = float(row["FixedCost"])
    c_ij[(i, j)] = float(row["VariableCost"])
    p_j[j] = float(row["profit_j"])

B_val = float(df["B"].iloc[0])
theta_val = float(df["θ"].iloc[0])

# Compute M_ij = min(a_i, b_j)
M_ij = {(i, j): min(a_i[i], b_j[j]) for i in range(m) for j in range(n)}

# Check supply-demand balance
sum_supply = sum(a_i.values())
sum_demand = sum(b_j.values())
if abs(sum_supply - sum_demand) > 1e-6:
    print(f"⚠️ Warning: total supply ({sum_supply}) ≠ total demand ({sum_demand}).")

# ---------------------------------------------------------------
# 3. BUILD LP MODEL
# ---------------------------------------------------------------
model = ConcreteModel()
model.I = RangeSet(0, m - 1)
model.J = RangeSet(0, n - 1)

# Parameters
model.a = Param(model.I, initialize=a_i)
model.b = Param(model.J, initialize=b_j)
model.f = Param(model.I, model.J, initialize=f_ij)
model.c = Param(model.I, model.J, initialize=c_ij)
model.p = Param(model.J, initialize=p_j)
model.M = Param(model.I, model.J, initialize=M_ij)

# Variables
model.x = Var(model.I, model.J, within=NonNegativeReals)     # flow
model.y = Var(model.I, model.J, bounds=(0, 1))               # fractional facility usage (LP relax)

# Objective: Maximize total revenue - total cost
def obj_rule(mdl):
    revenue = sum(mdl.p[j] * mdl.x[i, j] for i in mdl.I for j in mdl.J)
    total_fixed = sum(mdl.f[i, j] * mdl.y[i, j] for i in mdl.I for j in mdl.J)
    total_var = sum(mdl.c[i, j] * mdl.x[i, j] for i in mdl.I for j in mdl.J)
    return revenue - (total_fixed + total_var)

model.obj = Objective(rule=obj_rule, sense=maximize)

# Constraints
# (1) Supply: total shipped ≤ supply
def supply_rule(mdl, i):
    return sum(mdl.x[i, j] for j in mdl.J) <= mdl.a[i]
model.supply = Constraint(model.I, rule=supply_rule)

# (2) Demand: total received ≤ demand
def demand_rule(mdl, j):
    return sum(mdl.x[i, j] for i in mdl.I) <= mdl.b[j]
model.demand = Constraint(model.J, rule=demand_rule)

# (3) Linking: x_ij ≤ M_ij * y_ij
def linking_rule(mdl, i, j):
    return mdl.x[i, j] <= mdl.M[i, j] * mdl.y[i, j]
model.linking = Constraint(model.I, model.J, rule=linking_rule)

# ---------------------------------------------------------------
# 4. SOLVE MODEL
# ---------------------------------------------------------------
solver = SolverFactory("cplex")  # Change to 'cbc' if CPLEX not available
print("\n🚀 Solving LP-relaxation with CPLEX...")
results = solver.solve(model, tee=True)

# ---------------------------------------------------------------
# 5. EXTRACT RESULTS
# ---------------------------------------------------------------
opt_value = value(model.obj)

rows = []
for i in range(m):
    for j in range(n):
        rows.append({
            "Source": sources[i],
            "Sink": sinks[j],
            "FixedCost": float(model.f[i, j]),
            "VariableCost": float(model.c[i, j]),
            "a_i": float(model.a[i]),
            "b_j": float(model.b[j]),
            "profit_j": float(model.p[j]),
            "M_ij": float(model.M[i, j]),
            "B": B_val,
            "θ": theta_val,
            "x_ij": round(value(model.x[i, j]), 6),
            "y_ij": round(value(model.y[i, j]), 6),
            "OptimalValue": round(opt_value, 6)
        })

df_solution = pd.DataFrame(rows)

# Summary
total_revenue = sum(float(model.p[j]) * sum(value(model.x[i, j]) for i in model.I) for j in model.J)
total_cost_fixed = sum(float(model.f[i, j]) * value(model.y[i, j]) for i in model.I for j in model.J)
total_cost_variable = sum(float(model.c[i, j]) * value(model.x[i, j]) for i in model.I for j in model.J)
total_cost = total_cost_fixed + total_cost_variable

df_summary = pd.DataFrame({
    "Model": ["LP_relaxation"],
    "OptimalValue": [round(opt_value, 6)],
    "TotalRevenue": [round(total_revenue, 6)],
    "TotalFixedCost": [round(total_cost_fixed, 6)],
    "TotalVariableCost": [round(total_cost_variable, 6)],
    "TotalCost": [round(total_cost, 6)]
})

# ---------------------------------------------------------------
# 6. WRITE OUTPUT TO EXCEL
# ---------------------------------------------------------------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = f"FCTP_LP_Relax_Solution_{ts}.xlsx"

with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Input", index=False)
    df_solution.to_excel(writer, sheet_name="Solution", index=False)
    df_summary.to_excel(writer, sheet_name="Summary", index=False)

print(f"\n✅ LP-relaxation results saved to: {out_file}")
print("\n=== LP-relaxation Summary ===")
print(df_summary.to_string(index=False))
