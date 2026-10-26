# fctp_IP_single_supply_FC.py
# ---------------------------------------------------------------
# Solves one Fixed Charge Transportation Problem (FCTP) instance
# for a SINGLE SUPPLY NODE and MANY DEMAND NODES
# from a single-sheet Excel file.
#
# Objective: Maximize Total Revenue = (Total Profit from shipped goods)
#             - (Fixed + Variable Transportation Costs)
#
# This version:
#   - Assumes 1 supply node (capacity a) and many demand nodes j in J
#   - Builds a base model ("before")
#   - Builds an "after" model with FLOW COVER INEQUALITIES
#     automatically generated for the supply node.
#
# Outputs:
#   - Prints IP, LP, and Gap% before & after
#   - Saves Excel file containing:
#       "Summary",
#       "IP_before", "LP_before",
#       "IP_after",  "LP_after"
# ---------------------------------------------------------------

import pandas as pd
import time
from itertools import combinations
from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, NonNegativeReals, Binary,
    Constraint, Objective, SolverFactory, value, maximize
)

# -------------------------------
# SETTINGS
# -------------------------------
input_file = "Input_data_3_supply.xlsx"   # should contain only one Source (e.g., S1)
output_file = "FCTP_gap_3_supply.xlsx"
solver_name = "cbc"
show_solver_output = False
# -------------------------------


# ---------- MODEL BUILDER (1 supply, many demands) ----------
def build_model_from_df(df, lp_relax=False, add_flowcover=False):
    """
    Build Pyomo model for:
        - 1 supply node with capacity a
        - many demand nodes j in J

    Excel columns expected:
        Source, Sink, FixedCost, VariableCost, a_i, b_j, profit_j, ...

    All rows must have the same 'Source' and same 'a_i' (supply).
    """
    # Unique demand nodes
    sinks = df["Sink"].unique().tolist()
    n = len(sinks)
    sink_map = {t: j for j, t in enumerate(sinks)}

    # Single supply capacity a (assumed same in all rows)
    a_val = float(df["a_i"].iloc[0])

    # Demand b_j per sink
    b_j_series = df.groupby("Sink")["b_j"].first()
    b_j = {j: float(b_j_series[sinks[j]]) for j in range(n)}

    # Parameters per demand node
    f_j, c_j, p_j = {}, {}, {}
    for _, row in df.iterrows():
        j = sink_map[row["Sink"]]
        f_j[j] = float(row["FixedCost"])
        c_j[j] = float(row["VariableCost"])
        p_j[j] = float(row["profit_j"])

    # Big-M for each demand: min(supply, demand)
    M_j = {j: min(a_val, b_j[j]) for j in range(n)}

    # ---- Pyomo model ----
    model = ConcreteModel()
    model.J = RangeSet(0, n - 1)

    model.a = Param(initialize=a_val)
    model.b = Param(model.J, initialize=b_j)
    model.f = Param(model.J, initialize=f_j)
    model.c = Param(model.J, initialize=c_j)
    model.p = Param(model.J, initialize=p_j)
    model.M = Param(model.J, initialize=M_j)

    # Variables: x_j, y_j
    model.x = Var(model.J, within=NonNegativeReals)
    model.y = Var(
        model.J,
        within=Binary if not lp_relax else NonNegativeReals,
        bounds=(0, 1) if lp_relax else None
    )

    # Objective
    def obj_rule(mdl):
        total_profit = sum(mdl.p[j] * mdl.x[j] for j in mdl.J)
        total_cost = sum(mdl.f[j] * mdl.y[j] + mdl.c[j] * mdl.x[j] for j in mdl.J)
        return total_profit - total_cost

    model.obj = Objective(rule=obj_rule, sense=maximize)

    # Supply constraint (single supply node)
    def supply_rule(mdl):
        return sum(mdl.x[j] for j in mdl.J) <= mdl.a

    model.supply = Constraint(rule=supply_rule)

    # Demand constraints
    def demand_rule(mdl, j):
        return mdl.x[j] <= mdl.b[j]

    model.demand = Constraint(model.J, rule=demand_rule)

    # Linking constraints: x_j <= M_j * y_j
    def linking_rule(mdl, j):
        return mdl.x[j] <= mdl.M[j] * mdl.y[j]

    model.linking = Constraint(model.J, rule=linking_rule)

   # --- FLOW COVER INEQUALITIES (for the single supply node) ---
    if add_flowcover:
        # We generate minimal covers C1 ⊆ J with:
        #   sum_{j in C1} M_j > a
        #   and no proper subset has sum > a (minimality)
        positive_js = [j for j in range(n) if M_j[j] > 0]

        covers = []
        for r in range(1, len(positive_js) + 1):
            for subset in combinations(positive_js, r):
                total_M = sum(M_j[j] for j in subset)
                if total_M <= a_val:
                    continue
                # check minimality
                is_minimal = True
                for k in range(1, r):
                    for sub2 in combinations(subset, k):
                        if sum(M_j[j] for j in sub2) > a_val:
                            is_minimal = False
                            break
                    if not is_minimal:
                        break
                if is_minimal:
                    covers.append(subset)
            # if we found covers of this smallest size, stop increasing r
            if covers:
                break

        # Add one flow-cover constraint per minimal cover found
        # General form for 1-supply node:
        #   sum_{j in C1} x_j + sum_{j in C1} (M_j - λ)^+ (1 - y_j) <= a
        #   where λ = sum_{j in C1} M_j - a
        for idx, C1 in enumerate(covers, start=1):
            total_M = sum(M_j[j] for j in C1)
            lam = total_M - a_val  # excess λ
            gamma = {j: max(M_j[j] - lam, 0.0) for j in C1}

            expr = (
                sum(model.x[j] for j in C1)
                + sum(gamma[j] * (1 - model.y[j]) for j in C1)
                <= model.a
            )
            setattr(model, f"flowcover_{idx}", Constraint(expr=expr))

    return model, sinks, sink_map



# ---------- SOLVER ----------
def solve_model(model, solver_name="cbc", tee=False):
    solver = SolverFactory("cbc")
    t0 = time.time()
    res = solver.solve(model, tee=tee)
    t1 = time.time()
    return {"opt_val": value(model.obj), "time": round(t1 - t0, 3)}


# ---------- SOLUTION EXTRACTOR ----------
def extract_solution(df, model, sinks, sink_map):
    df_sol = df.copy()
    df_sol["x_ij"] = df_sol.apply(
        lambda row: value(model.x[sink_map[row["Sink"]]]),
        axis=1
    )
    df_sol["y_ij"] = df_sol.apply(
        lambda row: value(model.y[sink_map[row["Sink"]]]),
        axis=1
    )
    return df_sol


# ---------- MAIN ----------
df = pd.read_excel(input_file)
print(f"\n📘 Solving 1-supply FCTP instance from file: {input_file}")
print("➡ Base model: no flow-cover inequalities.")
print("➡ After model: flow-cover inequalities ADDED for the single supply node.\n")

# --- BEFORE Flow Cover (base model) ---
model_ip_b, sinks_b, sink_map_b = build_model_from_df(
    df, lp_relax=False, add_flowcover=False
)
model_lp_b, _, sink_map_lp_b = build_model_from_df(
    df, lp_relax=True, add_flowcover=False
)
res_ip_b = solve_model(model_ip_b, solver_name, show_solver_output)
res_lp_b = solve_model(model_lp_b, solver_name, show_solver_output)
gap_before = 100 * (res_lp_b["opt_val"] - res_ip_b["opt_val"]) / res_ip_b["opt_val"]

# --- AFTER Flow Cover (flow-cover ON) ---
model_ip_a, sinks_a, sink_map_a = build_model_from_df(
    df, lp_relax=False, add_flowcover=True
)
model_lp_a, _, sink_map_lp_a = build_model_from_df(
    df, lp_relax=True, add_flowcover=True
)
res_ip_a = solve_model(model_ip_a, solver_name, show_solver_output)
res_lp_a = solve_model(model_lp_a, solver_name, show_solver_output)
gap_after = 100 * (res_lp_a["opt_val"] - res_ip_a["opt_val"]) / res_ip_a["opt_val"]

# --- Extract Solutions ---
df_ip_before = extract_solution(df, model_ip_b, sinks_b, sink_map_b)
df_lp_before = extract_solution(df, model_lp_b, sinks_b, sink_map_lp_b)
df_ip_after = extract_solution(df, model_ip_a, sinks_a, sink_map_a)
df_lp_after = extract_solution(df, model_lp_a, sinks_a, sink_map_lp_a)

# ---------- PRINT RESULTS ----------
print("=== GAP COMPARISON (1 supply, with flow cover in AFTER) ===")
print(f"IP Objective (before):  {res_ip_b['opt_val']:.4f}")
print(f"LP Objective (before):  {res_lp_b['opt_val']:.4f}")
print(f"Gap Before:             {gap_before:.4f}%\n")

print(f"IP Objective (after):   {res_ip_a['opt_val']:.4f}")
print(f"LP Objective (after):   {res_lp_a['opt_val']:.4f}")
print(f"Gap After:              {gap_after:.4f}%")
print(f"💡 Gap Reduction         {gap_before - gap_after:.4f}%\n")

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
            "Gap Reduction%",
            "Flow cover in BEFORE?",
            "Flow cover in AFTER?",
            "Number of demand nodes"
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
            gap_before - gap_after,
            "No",
            "Yes (minimal covers for supply node)",
            len(sinks_b)
        ]
    })
    summary.to_excel(writer, sheet_name="Summary", index=False)

    df_ip_before.to_excel(writer, sheet_name="IP_before", index=False)
    df_lp_before.to_excel(writer, sheet_name="LP_before", index=False)
    df_ip_after.to_excel(writer, sheet_name="IP_after", index=False)
    df_lp_after.to_excel(writer, sheet_name="LP_after", index=False)

print(f"✅ Results saved to: {output_file}")