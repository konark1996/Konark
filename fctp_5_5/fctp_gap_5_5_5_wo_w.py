# fctp_IP_single.py
# ---------------------------------------------------------------
# Solves one Fixed Charge Transportation Problem (FCTP) instance
# from a single-sheet Excel file.
#
# Objective: Maximize Total Revenue = (Total Profit from shipped goods)
#             - (Fixed + Variable Transportation Costs)
#
# Outputs:
#   - Prints IP, LP, and Gap% before & after flow cover inequalities
#   - Saves Excel file (with inequalities) containing:
#       "Summary",
#       "IP_before", "LP_before",
#       "IP_after",  "LP_after"
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
input_file = "Input_data_5_5_5.xlsx"
output_file = "fctp_gap_5_5_5_wo_w.xlsx"
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
    model.y = Var(
        model.I,
        model.J,
        within=Binary if not lp_relax else NonNegativeReals,
        bounds=(0, 1) if lp_relax else None
    )

    # Objective
    def obj_rule(mdl):
        total_profit = sum(mdl.p[j] * mdl.x[i, j] for i in mdl.I for j in mdl.J)
        total_cost = sum(
            mdl.f[i, j] * mdl.y[i, j] + mdl.c[i, j] * mdl.x[i, j]
            for i in mdl.I for j in mdl.J
        )
        return total_profit - total_cost

    model.obj = Objective(rule=obj_rule, sense=maximize)

    # Constraints
    model.supply = Constraint(
        model.I,
        rule=lambda mdl, i: sum(mdl.x[i, j] for j in mdl.J) <= mdl.a[i]
    )
    model.demand = Constraint(
        model.J,
        rule=lambda mdl, j: sum(mdl.x[i, j] for i in mdl.I) <= mdl.b[j]
    )
    model.linking = Constraint(
        model.I, model.J,
        rule=lambda mdl, i, j: mdl.x[i, j] <= mdl.M[i, j] * mdl.y[i, j]
    )

    # # --- FLOW COVER INEQUALITIES (ONLY WHEN add_flowcover=True) ---
    if add_flowcover:
# (1) Iteration 1 : Supply node i = 0, arcs (0,1),(0,4)
        model.flowcover1 = Constraint(
            expr = model.x[0,1] + model.x[0,4]
            - 3*model.y[0,1]
            - 1*model.y[0,4] <= 4
        )

# (2) Iteration 1 : Supply node i = 3, arcs (3,2),(3,4)
        model.flowcover2 = Constraint(
            expr = model.x[3,2] + model.x[3,4]
            - 2*model.y[3,2]
            - 1*model.y[3,4] <= 4
        )

# (3) Iteration 1 : Demand node j = 4, arcs (0,4),(2,4)
        model.flowcover3 = Constraint(
            expr = model.x[0,4] + model.x[2,4]
            - 2*model.y[0,4]
            <= 3
        )

# (4) Iteration 2 : Supply node i = 0, arcs (0,1),(0,2)
        model.flowcover4 = Constraint(
            expr = model.x[0,1] + model.x[0,2]
            - 2*model.y[0,1]
            - 1*model.y[0,2] <= 5
        )

        model.flowcover5 = Constraint(
            expr=0.1*model.x[0, 1] + 0.1*model.x[0, 2] + 0.1*model.x[0, 4]
                +0.1*model.x[2, 1] + 0.1*model.x[2, 2] + 0.1*model.x[2, 4]
                +0.1*model.x[3, 1] + 0.1*model.x[3, 2] + 0.1*model.x[3, 4]
                - 0.2*model.y[0, 1] - 0.2*model.y[0, 2] - 0.2*model.y[0, 4]
                -0.2*model.x[2,1] - 0.2*model.x[2,2] - 0.2*model.x[2,4]
                -0.2*model.x[3, 1] - 0.2*model.x[3, 2] - 0.2*model.x[3, 4] <= 1        
        )


        #         # (6) For arcs (0,1) and (0,2) ,(0,3)
        # model.flowcover6 = Constraint(
        #     expr=0.25*model.x[0, 0] + 0.25*model.x[0, 1] + 0.25*model.x[0, 2] + 
        #          0.25*model.x[2, 0] + 0.25*model.x[2, 1] + 0.25*model.x[2, 2]
        #          - 0.583*model.y[0, 0] - 0.583*model.y[0, 1] - 0.583*model.y[0, 2] 
        #          - 0.583*model.y[2, 0] - 0.583*model.y[2, 1] - 0.583*model.y[2, 2]   <= 1
        # )

    return model, sources, sinks, source_map, sink_map


# ---------- SOLVER ----------
def solve_model(model, solver_name="cplex", tee=False):
    solver = SolverFactory(solver_name)
    model.write("fctp.lp", io_options={"symbolic_solver_labels": True})
    t0 = time.time()
    res = solver.solve(model, tee=tee)
    t1 = time.time()
    return {"opt_val": value(model.obj), "time": round(t1 - t0, 3)}


# ---------- SOLUTION EXTRACTOR ----------
def extract_solution(df, model, sources, sinks, source_map, sink_map):
    df_sol = df.copy()
    df_sol["x_ij"] = df_sol.apply(
        lambda row: value(model.x[source_map[row["Source"]],
                                  sink_map[row["Sink"]]]),
        axis=1
    )
    df_sol["y_ij"] = df_sol.apply(
        lambda row: value(model.y[source_map[row["Source"]],
                                  sink_map[row["Sink"]]]),
        axis=1
    )
    return df_sol


# ---------- MAIN ----------
df = pd.read_excel(input_file)
print(f"\n📘 Solving FCTP instance from file: {input_file}")

# --- BEFORE Flow Cover ---
model_ip_b, sources_b, sinks_b, source_map_b, sink_map_b = build_model_from_df(
    df, lp_relax=False, add_flowcover=False
)
model_lp_b, _, _, source_map_lp_b, sink_map_lp_b = build_model_from_df(
    df, lp_relax=True, add_flowcover=False
)
res_ip_b = solve_model(model_ip_b, solver_name, show_solver_output)
res_lp_b = solve_model(model_lp_b, solver_name, show_solver_output)
gap_before = 100 * (res_lp_b["opt_val"] - res_ip_b["opt_val"]) / res_ip_b["opt_val"]

# --- AFTER Flow Cover ---
model_ip_a, sources_a, sinks_a, source_map_a, sink_map_a = build_model_from_df(
    df, lp_relax=False, add_flowcover=True
)
model_lp_a, _, _, source_map_lp_a, sink_map_lp_a = build_model_from_df(
    df, lp_relax=True, add_flowcover=True
)
res_ip_a = solve_model(model_ip_a, solver_name, show_solver_output)
res_lp_a = solve_model(model_lp_a, solver_name, show_solver_output)
gap_after = 100 * (res_lp_a["opt_val"] - res_ip_a["opt_val"]) / res_ip_a["opt_val"]

# --- Extract Solutions (before and after) ---
df_ip_before = extract_solution(
    df, model_ip_b, sources_b, sinks_b, source_map_b, sink_map_b
)
df_lp_before = extract_solution(
    df, model_lp_b, sources_b, sinks_b, source_map_lp_b, sink_map_lp_b
)
df_ip_after = extract_solution(
    df, model_ip_a, sources_a, sinks_a, source_map_a, sink_map_a
)
df_lp_after = extract_solution(
    df, model_lp_a, sources_a, sinks_a, source_map_lp_a, sink_map_lp_a
)

# ---------- PRINT RESULTS ----------
print("\n=== GAP COMPARISON ===")
print(f"Before Flow Cover: {gap_before:.2f}%")
print(f"After Flow Cover : {gap_after:.2f}%")
print(f"💡 Gap Reduction   : {gap_before - gap_after:.2f}%")

# --- SAVE MODELS TO .lp FILES ONLY ---
model_ip_b.write("IP_before.lp", io_options={"symbolic_solver_labels": True})
model_lp_b.write("LP_before.lp", io_options={"symbolic_solver_labels": True})

model_ip_a.write("IP_after.lp", io_options={"symbolic_solver_labels": True})
model_lp_a.write("LP_after.lp", io_options={"symbolic_solver_labels": True})


# ---------- SAVE TO EXCEL ----------
with pd.ExcelWriter(output_file) as writer:
    summary = pd.DataFrame({
        "Metric": [
            "IP Objective (before)",
            "LP Objective (before)",
            "Gap% (before)",
            "IP Time (before, s)",
            "LP Time (before, s)",
            "IP Objective (after)",
            "LP Objective (after)",
            "Gap% (after)",
            "IP Time (after, s)",
            "LP Time (after, s)",
            "Gap Before%",
            "Gap After%",
            "Gap Reduction%"
        ],
        "Value": [
            res_ip_b["opt_val"],
            res_lp_b["opt_val"],
            gap_before,
            res_ip_b["time"],
            res_lp_b["time"],
            res_ip_a["opt_val"],
            res_lp_a["opt_val"],
            gap_after,
            res_ip_a["time"],
            res_lp_a["time"],
            gap_before,
            gap_after,
            gap_before - gap_after
        ]
    })
    summary.to_excel(writer, sheet_name="Summary", index=False)

    # 1st two sheets: solutions WITHOUT flow-cover inequalities
    df_ip_before.to_excel(writer, sheet_name="IP_before", index=False)
    df_lp_before.to_excel(writer, sheet_name="LP_before", index=False)

    # Next two sheets: solutions WITH flow-cover inequalities
    df_ip_after.to_excel(writer, sheet_name="IP_after", index=False)
    df_lp_after.to_excel(writer, sheet_name="LP_after", index=False)

print(f"\n✅ Results saved to: {output_file}")

