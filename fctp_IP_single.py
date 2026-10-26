# fctp_IP_single.py
# ---------------------------------------------------------------
# Solves one Fixed Charge Transportation Problem (FCTP) instance
# from a single-sheet Excel file.
#
# Objective: Maximize Total Revenue = (Total Profit from shipped goods)
#             - (Fixed + Variable Transportation Costs)
#
# Prints:
#   - IP Objective (Total Revenue)
#   - LP Relaxation Objective (Total Revenue)
#   - Optimality Gap%
# ---------------------------------------------------------------

import pandas as pd
import time
from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, NonNegativeReals, Binary,
    Constraint, Objective, SolverFactory, value, maximize
)

# -------------------------------
# SETTINGS
# -------------------------------
input_file = "Input_data_1.xlsx"   # Your Excel file
solver_name = "cplex"              # Change to "cbc" if CPLEX unavailable
show_solver_output = False         # Set True to view solver logs
# -------------------------------


# ---------- Model Builder ----------
def build_model_from_df(df, lp_relax=False):
    sources = df["Source"].unique().tolist()
    sinks = df["Sink"].unique().tolist()
    m, n = len(sources), len(sinks)
    source_map = {s: i for i, s in enumerate(sources)}
    sink_map = {t: j for j, t in enumerate(sinks)}

    # Supply and Demand
    a_i_series = df.groupby("Source")["a_i"].first()
    b_j_series = df.groupby("Sink")["b_j"].first()
    a_i = {i: float(a_i_series[sources[i]]) for i in range(m)}
    b_j = {j: float(b_j_series[sinks[j]]) for j in range(n)}

    # Parameters
    f_ij, c_ij, p_j = {}, {}, {}
    for _, row in df.iterrows():
        i, j = source_map[row["Source"]], sink_map[row["Sink"]]
        f_ij[(i, j)] = float(row["FixedCost"])
        c_ij[(i, j)] = float(row["VariableCost"])
        p_j[j] = float(row["profit_j"])

    # Big-M
    M_ij = {(i, j): min(a_i[i], b_j[j]) for i in range(m) for j in range(n)}

    model = ConcreteModel()
    model.I = RangeSet(0, m - 1)
    model.J = RangeSet(0, n - 1)

    model.a = Param(model.I, initialize=a_i)
    model.b = Param(model.J, initialize=b_j)
    model.f = Param(model.I, model.J, initialize=f_ij)
    model.c = Param(model.I, model.J, initialize=c_ij)
    model.p = Param(model.J, initialize=p_j)
    model.M = Param(model.I, model.J, initialize=M_ij)

    model.x = Var(model.I, model.J, within=NonNegativeReals)
    model.y = Var(model.I, model.J,
                  within=Binary if not lp_relax else NonNegativeReals,
                  bounds=(0, 1) if lp_relax else None)

    # Objective: Maximize Total Revenue
    def obj_rule(mdl):
        total_profit = sum(mdl.p[j] * mdl.x[i, j] for i in mdl.I for j in mdl.J)
        total_cost = sum(mdl.f[i, j] * mdl.y[i, j] + mdl.c[i, j] * mdl.x[i, j]
                         for i in mdl.I for j in mdl.J)
        return total_profit - total_cost

    model.obj = Objective(rule=obj_rule, sense=maximize)

    # Constraints
    model.supply = Constraint(model.I, rule=lambda mdl, i: sum(mdl.x[i, j] for j in mdl.J) <= mdl.a[i])
    model.demand = Constraint(model.J, rule=lambda mdl, j: sum(mdl.x[i, j] for i in mdl.I) <= mdl.b[j])
    model.linking = Constraint(model.I, model.J, rule=lambda mdl, i, j: mdl.x[i, j] <= mdl.M[i, j] * mdl.y[i, j])

    return model


# ---------- Solver ----------
def solve_model(model, solver_name="cplex", tee=False):
    solver = SolverFactory(solver_name)
    t0 = time.time()
    res = solver.solve(model, tee=tee)
    t1 = time.time()

    return {
        "opt_val": value(model.obj),
        "time": round(t1 - t0, 3)
    }


# ---------- MAIN EXECUTION ----------
df = pd.read_excel(input_file)
print(f"\n📘 Solving FCTP instance from file: {input_file}")

# Solve IP and LP
model_ip = build_model_from_df(df, lp_relax=False)
res_ip = solve_model(model_ip, solver_name, show_solver_output)

model_lp = build_model_from_df(df, lp_relax=True)
res_lp = solve_model(model_lp, solver_name, show_solver_output)

# Compute Gap%
gap = 100 * (res_lp["opt_val"] - res_ip["opt_val"]) / res_lp["opt_val"] if res_lp["opt_val"] != 0 else None

print("\n=== RESULTS ===")
print(f"IP Total Revenue = {res_ip['opt_val']:.2f}")
print(f"LP Total Revenue = {res_lp['opt_val']:.2f}")
print(f"Optimality Gap% = {gap:.2f}%")
print("=============================\n")
