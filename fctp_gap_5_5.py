# fctp_batch_solve.py
# ---------------------------------------------------------------
# Batch solver for multiple Fixed Charge Transportation Problem (FCTP) instances.
# Each sheet in the input Excel workbook is one (B, θ) instance.
# Solves both:
#   - Integer Program (IP) with binary y
#   - LP Relaxation (LP) with continuous y ∈ [0,1]
#
# Objective: Maximize Total Revenue = (Total Profit from shipped goods)
#             - (Fixed + Variable Transportation Costs)
#
# Saves:
#   - "IP_<sheet>" and "LP_<sheet>" sheets with arc-level results
#   - A master "Summary" sheet with key performance metrics and gap%
# ---------------------------------------------------------------

import time
from datetime import datetime
import pandas as pd
from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, NonNegativeReals, Binary,
    Constraint, Objective, SolverFactory, value, maximize
)

# ---------- SETTINGS ----------
input_workbook = "FCTP_VC_0_5_5.xlsx"      # multi-sheet input file
output_workbook = f"FCTP_Batch_Solutions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
default_solver = "cplex"                # change to "cbc" if CPLEX not available
show_solver_output = False              # True → show solver log
# -------------------------------


def safe_sheet_name(name: str) -> str:
    """Truncate sheet name safely (Excel max 31 chars)."""
    return str(name)[:31]


# ---------- Build Model ----------
def build_model_from_df(df_instance, lp_relax=False):
    sources = list(pd.Categorical(df_instance["Source"], categories=df_instance["Source"].unique(), ordered=True).categories)
    sinks = list(pd.Categorical(df_instance["Sink"], categories=df_instance["Sink"].unique(), ordered=True).categories)
    m, n = len(sources), len(sinks)
    source_map = {s: i for i, s in enumerate(sources)}
    sink_map = {t: j for j, t in enumerate(sinks)}

    # Supply & demand
    a_i_series = df_instance.groupby("Source")["a_i"].first()
    b_j_series = df_instance.groupby("Sink")["b_j"].first()
    a_i = {i: float(a_i_series[sources[i]]) for i in range(m)}
    b_j = {j: float(b_j_series[sinks[j]]) for j in range(n)}

    # Parameters
    f_ij, c_ij, p_j = {}, {}, {}
    for _, row in df_instance.iterrows():
        i, j = source_map[row["Source"]], sink_map[row["Sink"]]
        f_ij[(i, j)] = float(row["FixedCost"])
        c_ij[(i, j)] = float(row["VariableCost"])
        p_j[j] = float(row["profit_j"])  # total profit per unit shipped

    # Big-M value for linking constraint
    M_ij = {(i, j): float(min(a_i[i], b_j[j])) for i in range(m) for j in range(n)}

    # Model
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
    model.y = Var(model.I, model.J, within=Binary if not lp_relax else NonNegativeReals,
                  bounds=(0, 1) if lp_relax else None)

    # Objective: Maximize Total Revenue = Total Profit - (Fixed + Variable Costs)
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

    model._sources, model._sinks = sources, sinks
    return model


# ---------- Solve & Extract ----------
def solve_and_extract(model, solver_name=default_solver, tee=show_solver_output):
    solver = SolverFactory(solver_name)
    t0 = time.time()
    try:
        res = solver.solve(model, tee=tee)
        status = str(getattr(res.solver, "status", "Unknown"))
        term = str(getattr(res.solver, "termination_condition", "Unknown"))
    except Exception as e:
        return {"opt_val": None, "status": f"ERROR: {e}", "arc_rows": [], "solve_time": None}

    t1 = time.time()
    solve_time = round(t1 - t0, 4)

    # Compute components
    total_revenue = value(model.obj)
    total_profit = sum(value(model.p[j]) * value(model.x[i, j]) for i in model.I for j in model.J)
    fixed_cost = sum(value(model.f[i, j]) * value(model.y[i, j]) for i in model.I for j in model.J)
    var_cost = sum(value(model.c[i, j]) * value(model.x[i, j]) for i in model.I for j in model.J)

    rows = []
    for i in model.I:
        for j in model.J:
            rows.append({
                "Source": model._sources[i],
                "Sink": model._sinks[j],
                "a_i": value(model.a[i]),
                "b_j": value(model.b[j]),
                "x_ij": value(model.x[i, j]),
                "y_ij": value(model.y[i, j]),
                "FixedCost": value(model.f[i, j]),
                "VariableCost": value(model.c[i, j]),
                "M_ij": value(model.M[i, j]),
                "TotalProfit_ij": value(model.p[j]) * value(model.x[i, j])
            })

    return {
        "opt_val": total_revenue,
        "total_profit": total_profit,
        "fixed_cost": fixed_cost,
        "var_cost": var_cost,
        "solve_time": solve_time,
        "status": status,
        "termination": term,
        "arc_rows": rows
    }


# ---------- Batch Solve ----------
xls = pd.ExcelFile(input_workbook)
input_sheets = xls.sheet_names
summary_rows, output_dataframes = [], {}

for sheet in input_sheets:
    print(f"\n=== Solving instance: {sheet} ===")
    df_in = pd.read_excel(input_workbook, sheet_name=sheet)

    req_cols = ["Source", "Sink", "FixedCost", "VariableCost", "a_i", "b_j", "profit_j", "B", "θ"]
    if not all(c in df_in.columns for c in req_cols):
        print(f"Skipping '{sheet}' (missing columns)")
        continue

    # Solve IP and LP
    model_ip = build_model_from_df(df_in, lp_relax=False)
    res_ip = solve_and_extract(model_ip)

    model_lp = build_model_from_df(df_in, lp_relax=True)
    res_lp = solve_and_extract(model_lp)

    df_ip = pd.DataFrame(res_ip["arc_rows"])
    df_lp = pd.DataFrame(res_lp["arc_rows"])
    df_ip["B"] = df_lp["B"] = df_in["B"].iloc[0]
    df_ip["θ"] = df_lp["θ"] = df_in["θ"].iloc[0]

    output_dataframes[safe_sheet_name(f"IP_{sheet}")] = df_ip
    output_dataframes[safe_sheet_name(f"LP_{sheet}")] = df_lp

    # Optimality Gap
    ip_opt, lp_opt = res_ip["opt_val"], res_lp["opt_val"]
    gap = 100 * (lp_opt - ip_opt) / ip_opt if (ip_opt and lp_opt and ip_opt != 0) else None

    summary_rows.append({
        "InputSheet": sheet,
        "B": df_in["B"].iloc[0],
        "θ": df_in["θ"].iloc[0],
        "IP_TotalRevenue": ip_opt,
        "LP_TotalRevenue": lp_opt,
        "Gap%(wrt_IP)": gap,
        "IP_TotalProfit": res_ip["total_profit"],
        "IP_FixedCost": res_ip["fixed_cost"],
        "IP_VarCost": res_ip["var_cost"],
        "LP_TotalProfit": res_lp["total_profit"],
        "LP_FixedCost": res_lp["fixed_cost"],
        "LP_VarCost": res_lp["var_cost"],
        "IP_Time(s)": res_ip["solve_time"],
        "LP_Time(s)": res_lp["solve_time"],
        "IP_Status": res_ip["status"],
        "LP_Status": res_lp["status"]
    })


# ---------- Write Outputs ----------
with pd.ExcelWriter(output_workbook, engine="openpyxl") as writer:
    # 1️⃣ Write the Summary first
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

    # 2️⃣ Then write input sheets
    for sheet in input_sheets:
        df_in = pd.read_excel(input_workbook, sheet_name=sheet)
        df_in.to_excel(writer, sheet_name=safe_sheet_name(f"Input_{sheet}"), index=False)

    # 3️⃣ Then write output (IP & LP) results
    for sname, df_out in output_dataframes.items():
        if df_out.empty:
            pd.DataFrame([{"Note": "No data or solve failed"}]).to_excel(writer, sheet_name=sname, index=False)
        else:
            df_out.to_excel(writer, sheet_name=sname, index=False)

df_summary = pd.DataFrame(summary_rows)
print(f"Max Gap% = {df_summary['Gap%(wrt_IP)'].max():.2f}%")

print(f"\n✅ All FCTP instances solved and saved to '{output_workbook}' (Summary sheet first)")
