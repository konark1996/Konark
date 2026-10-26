# ---------------------------------------------------------------
# Fixed Charge Transportation Problem (FCTP)
# Cutting Plane Algorithm with Flow Cover Inequalities
# SUPPLY + DEMAND separation
# WITH:
#  - Math-form Flow Cover Inequalities printed in terminal
#  - Iteration-wise LP–IP gap reporting
#  - Excel outputs SAME AS ORIGINAL
#  - Terminal output saved to .txt file
# ---------------------------------------------------------------

import pandas as pd
import time
import sys
import io
from datetime import datetime
from pyomo.environ import *
from pyomo.opt import SolverFactory

# -------------------------------
# SETTINGS
# -------------------------------
input_file = "fctp_Input_data_30_30.xlsx"
output_file = f"fctp_gap_30_30_sep.xlsx"
log_file = "fctp_gap_30_30_sep_log.txt"       # <-- NEW
solver_name = "cplex"
EPS = 1e-6
MAX_ITER = 20
# -------------------------------


# ===============================================================
# TEE CLASS — mirrors every print() to terminal + log buffer
# ===============================================================
class Tee:
    def __init__(self):
        self.terminal = sys.stdout
        self.buffer = io.StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)

    def flush(self):
        self.terminal.flush()

    def getvalue(self):
        return self.buffer.getvalue()

tee = Tee()
sys.stdout = tee
# ===============================================================


# ===============================================================
# MODEL BUILDER
# ===============================================================
def build_model(df, cuts, lp_relax=True):

    sources = df["Source"].unique().tolist()
    sinks = df["Sink"].unique().tolist()
    smap = {s: i for i, s in enumerate(sources)}
    dmap = {d: j for j, d in enumerate(sinks)}

    a = df.groupby("Source")["a_i"].first().to_dict()
    b = df.groupby("Sink")["b_j"].first().to_dict()

    f, c, p = {}, {}, {}
    for _, r in df.iterrows():
        i, j = smap[r["Source"]], dmap[r["Sink"]]
        f[(i, j)] = float(r["FixedCost"])
        c[(i, j)] = float(r["VariableCost"])
        p[j] = float(r["profit_j"])

    M = {(i, j): min(a[sources[i]], b[sinks[j]])
         for i in range(len(sources)) for j in range(len(sinks))}

    m = ConcreteModel()
    m.I = RangeSet(0, len(sources) - 1)
    m.J = RangeSet(0, len(sinks) - 1)

    m.a = Param(m.I, initialize={i: a[sources[i]] for i in m.I})
    m.b = Param(m.J, initialize={j: b[sinks[j]] for j in m.J})
    m.M = Param(m.I, m.J, initialize=M)
    m.f = Param(m.I, m.J, initialize=f)
    m.c = Param(m.I, m.J, initialize=c)
    m.p = Param(m.J, initialize=p)

    m.x = Var(m.I, m.J, within=NonNegativeReals)
    m.y = Var(m.I, m.J, within=Binary if not lp_relax else UnitInterval)

    m.obj = Objective(
        expr=sum(
            m.p[j] * m.x[i, j]
            - m.f[i, j] * m.y[i, j]
            - m.c[i, j] * m.x[i, j]
            for i in m.I for j in m.J
        ),
        sense=maximize
    )

    m.supply = Constraint(m.I, rule=lambda m, i: sum(m.x[i, j] for j in m.J) <= m.a[i])
    m.demand = Constraint(m.J, rule=lambda m, j: sum(m.x[i, j] for i in m.I) <= m.b[j])
    m.link = Constraint(m.I, m.J, rule=lambda m, i, j: m.x[i, j] <= m.M[i, j] * m.y[i, j])

    m.flowcover = ConstraintList()
    for cut in cuts:
        m.flowcover.add(cut(m))

    return m, smap, dmap


# ===============================================================
# SUPPLY NODE SEPARATION
# ===============================================================
def separate_supply(model_lp):
    violated = []

    for i in model_lp.I:

        kp = ConcreteModel()
        kp.J = list(model_lp.J)
        kp.z = Var(kp.J, within=Binary)

        kp.obj = Objective(
            expr=sum((1 - value(model_lp.y[i, j])) * kp.z[j] for j in kp.J),
            sense=minimize
        )

        kp.cap = Constraint(
            expr=sum(value(model_lp.M[i, j]) * kp.z[j] for j in kp.J)
                 >= value(model_lp.a[i]) + EPS
        )

        SolverFactory(solver_name).solve(kp, tee=False)

        C1 = [j for j in kp.J if value(kp.z[j]) > 0.5]
        if not C1:
            continue

        lam = sum(value(model_lp.M[i, j]) for j in C1) - value(model_lp.a[i])

        lhs = (
            sum(value(model_lp.x[i, j]) for j in C1)
            + sum((value(model_lp.M[i, j]) - lam)
                  * (1 - value(model_lp.y[i, j])) for j in C1)
        )

        if lhs > value(model_lp.a[i]) + EPS:

            print(f"\n[SUPPLY FLOW COVER] Node i = {i}")
            print("C1 =", C1)
            print("∑ x_ij + ∑ (M_ij − λ)(1 − y_ij) ≤ a_i")
            print(
                " + ".join([f"x({i},{j})" for j in C1]) +
                " + " +
                " + ".join([f"({value(model_lp.M[i,j])-lam:.2f})(1-y({i},{j}))" for j in C1]) +
                f" <= a({i})"
            )

            def cut_rule(m, C1=C1, i=i, lam=lam):
                return (
                    sum(m.x[i, j] for j in C1)
                    + sum((m.M[i, j] - lam) * (1 - m.y[i, j]) for j in C1)
                    <= m.a[i]
                )

            violated.append(cut_rule)

    return violated


# ===============================================================
# DEMAND NODE SEPARATION
# ===============================================================
def separate_demand(model_lp):
    violated = []

    for j in model_lp.J:

        kp = ConcreteModel()
        kp.I = list(model_lp.I)
        kp.z = Var(kp.I, within=Binary)

        kp.obj = Objective(
            expr=sum((1 - value(model_lp.y[i, j])) * kp.z[i] for i in kp.I),
            sense=minimize
        )

        kp.cap = Constraint(
            expr=sum(value(model_lp.M[i, j]) * kp.z[i] for i in kp.I)
                 >= value(model_lp.b[j]) + EPS
        )

        SolverFactory(solver_name).solve(kp, tee=False)

        C1 = [i for i in kp.I if value(kp.z[i]) > 0.5]
        if not C1:
            continue

        lam = sum(value(model_lp.M[i, j]) for i in C1) - value(model_lp.b[j])

        lhs = (
            sum(value(model_lp.x[i, j]) for i in C1)
            + sum((value(model_lp.M[i, j]) - lam)
                  * (1 - value(model_lp.y[i, j])) for i in C1)
        )

        if lhs > value(model_lp.b[j]) + EPS:

            print(f"\n[DEMAND FLOW COVER] Node j = {j}")
            print("C1 =", C1)
            print("∑ x_ij + ∑ (M_ij − λ)(1 − y_ij) ≤ b_j")
            print(
                " + ".join([f"x({i},{j})" for i in C1]) +
                " + " +
                " + ".join([f"({value(model_lp.M[i,j])-lam:.2f})(1-y({i},{j}))" for i in C1]) +
                f" <= b({j})"
            )

            def cut_rule(m, C1=C1, j=j, lam=lam):
                return (
                    sum(m.x[i, j] for i in C1)
                    + sum((m.M[i, j] - lam) * (1 - m.y[i, j]) for i in C1)
                    <= m.b[j]
                )

            violated.append(cut_rule)

    return violated


# ===============================================================
# SOLUTION EXTRACTION (UNCHANGED)
# ===============================================================
def extract_solution(df, model, smap, dmap):
    out = df.copy()
    out["x_ij"] = out.apply(
        lambda r: value(model.x[smap[r["Source"]], dmap[r["Sink"]]]), axis=1
    )
    out["y_ij"] = out.apply(
        lambda r: value(model.y[smap[r["Source"]], dmap[r["Sink"]]]), axis=1
    )
    return out


# ===============================================================
# MAIN CUTTING PLANE
# ===============================================================
df = pd.read_excel(input_file)
solver = SolverFactory(solver_name)

cuts = []
gap_log = []
iteration = 0
separation_time = 0.0
runtime_after_cuts = 0.0

print("\n📘 FCTP Cutting Plane with Flow Cover Inequalities")
print(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------- BEFORE CUTS ----------
model_lp_b, sm, dm = build_model(df, cuts, lp_relax=True)
model_ip_b, _, _ = build_model(df, cuts, lp_relax=False)

t0 = time.perf_counter()
solver.solve(model_lp_b)
solver.solve(model_ip_b)
runtime_without_cuts = time.perf_counter() - t0

lp_before = value(model_lp_b.obj)
ip_before = value(model_ip_b.obj)
gap_before = 100 * (lp_before - ip_before) / ip_before

gap_log.append((0, lp_before, ip_before, gap_before))

print(f"\nIteration 0 (Before cuts)")
print(f"LP = {lp_before:.4f}, IP = {ip_before:.4f}, Gap% = {gap_before:.2f}")

# ---------- CUT LOOP ----------
while True:
    iteration += 1
    print(f"\n--- Iteration {iteration} ---")

    model_lp, _, _ = build_model(df, cuts, lp_relax=True)
    lp_solve_start = time.perf_counter()
    solver.solve(model_lp)
    runtime_after_cuts += time.perf_counter() - lp_solve_start

    separation_start = time.perf_counter()
    new_cuts = []
    new_cuts += separate_supply(model_lp)
    new_cuts += separate_demand(model_lp)
    separation_time += time.perf_counter() - separation_start

    model_ip_tmp, _, _ = build_model(df, cuts + new_cuts, lp_relax=False)
    solver.solve(model_ip_tmp)

    lp_val = value(model_lp.obj)
    ip_val = value(model_ip_tmp.obj)
    gap = 100 * (lp_val - ip_val) / ip_val

    gap_log.append((iteration, lp_val, ip_val, gap))

    print(f"LP = {lp_val:.4f}, IP = {ip_val:.4f}, Gap% = {gap:.2f}")

    if not new_cuts or iteration >= MAX_ITER:
        break

    cuts.extend(new_cuts)

# ---------- AFTER CUTS ----------
model_lp_a, _, _ = build_model(df, cuts, lp_relax=True)
model_ip_a, _, _ = build_model(df, cuts, lp_relax=False)

solver.solve(model_lp_a)
solver.solve(model_ip_a)

lp_after = value(model_lp_a.obj)
ip_after = value(model_ip_a.obj)
gap_after = 100 * (lp_after - ip_after) / ip_after

print(f"\n✅ Results saved to: {output_file}")
print(f"Total Flow Cover Inequalities added: {len(cuts)}")
print(f"Runtime Without Cuts (LP + IP): {runtime_without_cuts:.3f} seconds")
print(f"Separation Time: {separation_time:.3f} seconds")
print(f"Runtime After Adding Cuts (LPs across iterations): {runtime_after_cuts:.3f} seconds")

print("\n📊 GAP EVOLUTION")
print("Iter | LP Obj | IP Obj | Gap %")
print("-" * 38)
for it, lpv, ipv, g in gap_log:
    print(f"{it:>4} | {lpv:>8.2f} | {ipv:>8.2f} | {g:>6.2f}")


# ===============================================================
# RESTORE stdout AND SAVE LOG FILE            <-- NEW BLOCK
# ===============================================================
sys.stdout = tee.terminal                     # restore normal stdout
full_log = tee.getvalue()

with open(log_file, "w", encoding="utf-8") as f:
    f.write(full_log)

print(f"📄 Terminal log saved to: {log_file}")
# ===============================================================


# ===============================================================
# SAVE EXCEL OUTPUTS (UNCHANGED)
# ===============================================================
with pd.ExcelWriter(output_file) as writer:

    summary = pd.DataFrame({
        "Metric": [
            "IP Objective (before)", "LP Objective (before)", "Gap% (before)",
            "IP Objective (after)", "LP Objective (after)", "Gap% (after)",
            "Gap Reduction%",
            "Runtime Without Cuts (LP + IP) (seconds)",
            "Separation Time (seconds)",
            "Runtime After Adding Cuts (LPs across iterations) (seconds)"
        ],
        "Value": [
            ip_before, lp_before, gap_before,
            ip_after, lp_after, gap_after,
            gap_before - gap_after,
            round(runtime_without_cuts, 3),
            round(separation_time, 3),
            round(runtime_after_cuts, 3)
        ]
    })
    summary.to_excel(writer, "Summary", index=False)

    extract_solution(df, model_lp_b, sm, dm).to_excel(writer, "LP_before", index=False)
    extract_solution(df, model_ip_b, sm, dm).to_excel(writer, "IP_before", index=False)
    extract_solution(df, model_lp_a, sm, dm).to_excel(writer, "LP_after", index=False)
    extract_solution(df, model_ip_a, sm, dm).to_excel(writer, "IP_after", index=False)