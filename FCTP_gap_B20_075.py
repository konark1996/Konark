# ---------------------------------------------------------------
# fctp_IP_single.py  (Updated version)
# Solves Fixed Charge Transportation Problem (FCTP) from Excel
# and prints summary of IP & LP solutions
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
input_file = "B20_075.xlsx"
output_file = "FCTP_gap_B20_075.xlsx"
solver_name = "cplex"
show_solver_output = False
# -------------------------------


# ---------- MODEL BUILDER ----------
def build_model_from_df(df, lp_relax=False, add_flowcover=False):
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

    # Objective: Maximize (Profit – Fixed – Variable costs)
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

    return model, sources, sinks, source_map, sink_map


# ---------- SOLVER ----------
def solve_model(model, solver_name="cplex", tee=False):
    solver = SolverFactory(solver_name)
    t0 = time.time()
    res = solver.solve(model, tee=tee)
    t1 = time.time()
    return {"opt_val": value(model.obj), "time": round(t1 - t0, 3)}


# ---------- SOLUTION EXTRACTOR ----------
def extract_solution(df, model, sources, sinks, source_map, sink_map):
    df_sol = df.copy()
    df_sol["x_ij"] = df_sol.apply(lambda row: value(model.x[source_map[row["Source"]], sink_map[row["Sink"]]]), axis=1)
    df_sol["y_ij"] = df_sol.apply(lambda row: value(model.y[source_map[row["Source"]], sink_map[row["Sink"]]]), axis=1)
    return df_sol


# ---------- MAIN ----------
df = pd.read_excel(input_file)
print(f"\n📘 Solving FCTP instance from file: {input_file}\n")

# ---- Solve IP ----
model_ip, sources, sinks, source_map, sink_map = build_model_from_df(df, lp_relax=False)
res_ip = solve_model(model_ip, solver_name, tee=show_solver_output)

# ---- Solve LP Relaxation ----
model_lp, sources_lp, sinks_lp, source_map_lp, sink_map_lp = build_model_from_df(df, lp_relax=True)
res_lp = solve_model(model_lp, solver_name, tee=show_solver_output)

# ---- Calculate Gap ----
gap = round(((res_lp["opt_val"] - res_ip["opt_val"]) / abs(res_ip["opt_val"])) * 100, 3) if res_ip["opt_val"] != 0 else 0

# ---- Extract Solutions ----
df_ip = extract_solution(df, model_ip, sources, sinks, source_map, sink_map)
df_lp = extract_solution(df, model_lp, sources_lp, sinks_lp, source_map_lp, sink_map_lp)

# ---- Print Summary ----
print("📊 ----- RESULTS SUMMARY -----")
print(f"IP Objective Value      : {res_ip['opt_val']:.3f}")
print(f"LP Objective Value      : {res_lp['opt_val']:.3f}")
print(f"Gap (%)                 : {gap:.3f}")
print(f"IP Solve Time (sec)     : {res_ip['time']}")
print(f"LP Solve Time (sec)     : {res_lp['time']}")
print("-------------------------------")

# ---- Save to Excel ----
with pd.ExcelWriter(output_file) as writer:
    summary = pd.DataFrame({
        "Metric": ["IP Objective", "LP Objective", "Gap%", "IP Time (s)", "LP Time (s)"],
        "Value": [res_ip["opt_val"], res_lp["opt_val"], gap, res_ip["time"], res_lp["time"]]
    })
    summary.to_excel(writer, sheet_name="Summary", index=False)
    df_ip.to_excel(writer, sheet_name="IP Solution", index=False)
    df_lp.to_excel(writer, sheet_name="LP Solution", index=False)

print(f"\n✅ Results saved to: {output_file}")
