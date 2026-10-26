# ---------------------------------------------------------------
# FCTP - Polar Duality Cutting Plane (Standalone)
# Algorithm:
#   1. Solve LP relaxation → get x̄_ij, ȳ_ij values
#   2. Build polar dual LP using arc-level (x̄,ȳ) as RHS coefficients
#      Constraints: πy·ȳ_ij + πx·x̄_ij <= 1  for each arc (i,j)
#      Objective:   max  πx·(Σx̄_ij) + πy·(Σȳ_ij)
#   3. Solve polar dual → get πx, πy
#   4. Add polar cut:  πx·(Σx_ij) + πy·(Σy_ij) <= 1  to LP
#   5. Solve IP with updated cuts → compute gap%
#   6. Repeat from step 1 until gap% does not decrease
# ---------------------------------------------------------------

import pandas as pd
import sys
import io
import time
from datetime import datetime
from pyomo.environ import *
from pyomo.opt import SolverFactory

# -------------------------------
# SETTINGS
# -------------------------------
input_file  = "Input_data_5_5_1.xlsx"
output_file = "fctp_polar_5_5_1.xlsx"
log_file    = "fctp_polar_5_5_1_log.txt"
solver_name = "cplex"
EPS         = 1e-6
MAX_ITER    = 20
# -------------------------------


# ===============================================================
# TEE — mirrors print() to terminal + log buffer
# ===============================================================
class Tee:
    def __init__(self):
        self.terminal = sys.stdout
        self.buffer   = io.StringIO()

    def write(self, msg):
        self.terminal.write(msg)
        self.buffer.write(msg)

    def flush(self):
        self.terminal.flush()

    def getvalue(self):
        return self.buffer.getvalue()

tee = Tee()
sys.stdout = tee


# ===============================================================
# MODEL BUILDER  (LP relaxation or MIP)
# ===============================================================
def build_model(df, polar_cuts, lp_relax=True):
    """
    polar_cuts : list of dicts  {pi_x: float, pi_y: float}
                 Each entry adds the constraint:
                     pi_x * sum(x_ij) + pi_y * sum(y_ij) <= 1
    """
    sources = df["Source"].unique().tolist()
    sinks   = df["Sink"].unique().tolist()
    smap    = {s: i for i, s in enumerate(sources)}
    dmap    = {d: j for j, d in enumerate(sinks)}

    a = df.groupby("Source")["a_i"].first().to_dict()
    b = df.groupby("Sink")["b_j"].first().to_dict()

    f, c, p = {}, {}, {}
    for _, r in df.iterrows():
        i, j       = smap[r["Source"]], dmap[r["Sink"]]
        f[(i, j)]  = float(r["FixedCost"])
        c[(i, j)]  = float(r["VariableCost"])
        p[j]       = float(r["profit_j"])

    M = {(i, j): min(a[sources[i]], b[sinks[j]])
         for i in range(len(sources)) for j in range(len(sinks))}

    m   = ConcreteModel()
    m.I = RangeSet(0, len(sources) - 1)
    m.J = RangeSet(0, len(sinks)   - 1)

    m.a = Param(m.I, initialize={i: a[sources[i]] for i in m.I})
    m.b = Param(m.J, initialize={j: b[sinks[j]]   for j in m.J})
    m.M = Param(m.I, m.J, initialize=M)
    m.f = Param(m.I, m.J, initialize=f)
    m.c = Param(m.I, m.J, initialize=c)
    m.p = Param(m.J,       initialize=p)

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

    m.supply = Constraint(m.I,
        rule=lambda m, i: sum(m.x[i, j] for j in m.J) <= m.a[i])
    m.demand = Constraint(m.J,
        rule=lambda m, j: sum(m.x[i, j] for i in m.I) <= m.b[j])
    m.link   = Constraint(m.I, m.J,
        rule=lambda m, i, j: m.x[i, j] <= m.M[i, j] * m.y[i, j])

    # ---------- Polar cuts (added to LP only) ----------
    m.polar_cuts = ConstraintList()
    for cut in polar_cuts:
        pi_x = cut["pi_x"]
        pi_y = cut["pi_y"]
        m.polar_cuts.add(
            pi_x * sum(m.x[i, j] for i in m.I for j in m.J)
            + pi_y * sum(m.y[i, j] for i in m.I for j in m.J)
            <= 1
        )

    return m, smap, dmap


# ===============================================================
# STEP 2+3 : POLAR DUAL LP
# ===============================================================
def solve_polar_dual(x_bar, y_bar):
    """
    x_bar : dict  {(i,j): x̄_ij}   from current LP relaxation
    y_bar : dict  {(i,j): ȳ_ij}   from current LP relaxation

    Polar dual LP
    ─────────────
    Variables : π_x >= 0,  π_y >= 0

    max   π_x · (Σ x̄_ij)  +  π_y · (Σ ȳ_ij)

    s.t.  π_y · ȳ_ij  +  π_x · x̄_ij  <=  1   ∀ (i,j)
          π_x, π_y >= 0

    Printed math form for each arc is shown in the terminal.
    """
    arcs      = list(x_bar.keys())
    X_total   = sum(x_bar.values())
    Y_total   = sum(y_bar.values())

    pd_model        = ConcreteModel()
    pd_model.pi_x   = Var(within=NonNegativeReals)
    pd_model.pi_y   = Var(within=NonNegativeReals)

    # Objective
    pd_model.obj = Objective(
        expr  = pd_model.pi_x * X_total + pd_model.pi_y * Y_total,
        sense = maximize
    )

    # Arc-level constraints  π_y·ȳ_ij + π_x·x̄_ij <= 1
    pd_model.arc_cuts = ConstraintList()

    print("\n  [POLAR DUAL] Arc constraints added:")
    for (i, j) in arcs:
        xv = x_bar[(i, j)]
        yv = y_bar[(i, j)]
        pd_model.arc_cuts.add(
            pd_model.pi_y * yv + pd_model.pi_x * xv <= 1
        )
        print(f"    {yv:.4f}·π_y  +  {xv:.4f}·π_x  <=  1     [arc ({i},{j})]")

    print(f"\n  [POLAR DUAL] Objective:  max  {X_total:.4f}·π_x  +  {Y_total:.4f}·π_y")

    SolverFactory(solver_name).solve(pd_model, tee=False)

    pi_x = value(pd_model.pi_x)
    pi_y = value(pd_model.pi_y)

    print(f"  [POLAR DUAL] Solution:   π_x = {pi_x:.6f},  π_y = {pi_y:.6f}")
    print(f"  [POLAR DUAL] Polar cut:  {pi_x:.6f}·Σx_ij  +  {pi_y:.6f}·Σy_ij  <=  1")

    return pi_x, pi_y


# ===============================================================
# SOLUTION EXTRACTION
# ===============================================================
def extract_solution(df, model, smap, dmap):
    out = df.copy()
    out["x_ij"] = out.apply(
        lambda r: value(model.x[smap[r["Source"]], dmap[r["Sink"]]]), axis=1)
    out["y_ij"] = out.apply(
        lambda r: value(model.y[smap[r["Source"]], dmap[r["Sink"]]]), axis=1)
    return out


# ===============================================================
# MAIN  — POLAR DUALITY CUTTING PLANE
# ===============================================================
df     = pd.read_excel(input_file)
solver = SolverFactory(solver_name)

polar_cuts = []          # accumulates {pi_x, pi_y} dicts
gap_log    = []
iteration  = 0

print("\n📘 FCTP — Polar Duality Cutting Plane")
print(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── BASELINE (no cuts) ─────────────────────────────────────────
m_lp0, sm, dm = build_model(df, [], lp_relax=True)
m_ip0, _,  _  = build_model(df, [], lp_relax=False)

t0 = time.time(); solver.solve(m_lp0); t_lp0 = time.time() - t0
t0 = time.time(); solver.solve(m_ip0); t_ip0 = time.time() - t0

lp0  = value(m_lp0.obj)
ip0  = value(m_ip0.obj)
gap0 = 100 * (lp0 - ip0) / abs(ip0)

gap_log.append((0, lp0, ip0, gap0))
print(f"\nIteration 0 (Baseline — no polar cuts)")
print(f"  LP  = {lp0:.4f}")
print(f"  IP  = {ip0:.4f}")
print(f"  Gap = {gap0:.2f}%")

prev_gap = gap0

# ── POLAR CUT LOOP ─────────────────────────────────────────────
while iteration < MAX_ITER:
    iteration += 1
    print(f"\n{'='*55}")
    print(f"  ITERATION {iteration}")
    print(f"{'='*55}")

    # STEP 1 — solve LP relaxation with current polar cuts
    m_lp, _, _ = build_model(df, polar_cuts, lp_relax=True)
    solver.solve(m_lp)
    lp_val = value(m_lp.obj)

    # Read x̄_ij, ȳ_ij from LP solution
    x_bar = {(i, j): value(m_lp.x[i, j]) for i in m_lp.I for j in m_lp.J}
    y_bar = {(i, j): value(m_lp.y[i, j]) for i in m_lp.I for j in m_lp.J}

    print(f"\nSTEP 1 — LP relaxation solved.   LP obj = {lp_val:.4f}")
    print("  x̄_ij values:")
    for (i, j), v in x_bar.items():
        print(f"    x̄({i},{j}) = {v:.4f}")
    print("  ȳ_ij values:")
    for (i, j), v in y_bar.items():
        print(f"    ȳ({i},{j}) = {v:.4f}")

    # STEP 2+3 — build & solve polar dual, get π_x, π_y
    print(f"\nSTEP 2+3 — Polar dual LP:")
    pi_x, pi_y = solve_polar_dual(x_bar, y_bar)

    # Check if polar cut is trivially inactive (both zeros)
    if abs(pi_x) < EPS and abs(pi_y) < EPS:
        print("\n  ⚠ Polar dual returned π_x = π_y = 0. Stopping.")
        break

    # STEP 4 — add polar cut to the pool
    new_cut = {"pi_x": pi_x, "pi_y": pi_y}
    polar_cuts.append(new_cut)

    print(f"\nSTEP 4 — Polar cut added to pool  (total cuts = {len(polar_cuts)})")

    # STEP 5 — solve LP + IP with updated cuts, compute gap
    m_lp_new, _, _ = build_model(df, polar_cuts, lp_relax=True)
    m_ip_new, _, _ = build_model(df, polar_cuts, lp_relax=False)

    solver.solve(m_lp_new)
    solver.solve(m_ip_new)

    lp_new  = value(m_lp_new.obj)
    ip_new  = value(m_ip_new.obj)
    gap_new = 100 * (lp_new - ip_new) / abs(ip_new)

    gap_log.append((iteration, lp_new, ip_new, gap_new))

    print(f"\nSTEP 5 — After adding polar cut:")
    print(f"  LP  = {lp_new:.4f}")
    print(f"  IP  = {ip_new:.4f}")
    print(f"  Gap = {gap_new:.2f}%")

    # STEP 6 — stopping criterion
    if gap_new >= prev_gap - EPS:
        print(f"\n  ✅ Gap did not decrease ({prev_gap:.2f}% → {gap_new:.2f}%). Stopping.")
        break

    print(f"  Gap improved: {prev_gap:.2f}% → {gap_new:.2f}%")
    prev_gap = gap_new

# ── FINAL STATE ────────────────────────────────────────────────
m_lp_f, _, _ = build_model(df, polar_cuts, lp_relax=True)
m_ip_f, _, _ = build_model(df, polar_cuts, lp_relax=False)

t0 = time.time(); solver.solve(m_lp_f); t_lp_f = time.time() - t0
t0 = time.time(); solver.solve(m_ip_f); t_ip_f = time.time() - t0

lp_f  = value(m_lp_f.obj)
ip_f  = value(m_ip_f.obj)
gap_f = 100 * (lp_f - ip_f) / abs(ip_f)

print(f"\n{'='*55}")
print(f"FINAL RESULTS")
print(f"{'='*55}")
print(f"  Total polar cuts added : {len(polar_cuts)}")
print(f"  LP before              : {lp0:.4f}")
print(f"  LP after               : {lp_f:.4f}")
print(f"  IP before              : {ip0:.4f}")
print(f"  IP after               : {ip_f:.4f}")
print(f"  Gap before             : {gap0:.2f}%")
print(f"  Gap after              : {gap_f:.2f}%")
print(f"  Gap reduction          : {gap0 - gap_f:.2f}%")

print("\n📊 GAP EVOLUTION")
print(f"{'Iter':>4} | {'LP Obj':>10} | {'IP Obj':>10} | {'Gap %':>7}")
print("-" * 42)
for it, lpv, ipv, g in gap_log:
    print(f"{it:>4} | {lpv:>10.4f} | {ipv:>10.4f} | {g:>7.2f}")

# ── RESTORE stdout & SAVE LOG ──────────────────────────────────
sys.stdout = tee.terminal
full_log   = tee.getvalue()

with open(log_file, "w", encoding="utf-8") as f:
    f.write(full_log)
print(f"\n📄 Terminal log saved to: {log_file}")

# ── SAVE EXCEL ─────────────────────────────────────────────────
summary = pd.DataFrame({
    "Metric": [
        "LP Objective (before)", "IP Objective (before)", "Gap% (before)",
        "LP Time (before)",      "IP Time (before)",
        "LP Objective (after)",  "IP Objective (after)",  "Gap% (after)",
        "LP Time (after)",       "IP Time (after)",
        "Total Polar Cuts",      "Gap Reduction%"
    ],
    "Value": [
        lp0,            ip0,            round(gap0, 4),
        round(t_lp0,3), round(t_ip0,3),
        lp_f,           ip_f,           round(gap_f, 4),
        round(t_lp_f,3),round(t_ip_f,3),
        len(polar_cuts), round(gap0 - gap_f, 4)
    ]
})

gap_df = pd.DataFrame(gap_log, columns=["Iteration", "LP_Obj", "IP_Obj", "Gap%"])

cuts_df = pd.DataFrame([
    {"Cut #": k+1, "pi_x": c["pi_x"], "pi_y": c["pi_y"]}
    for k, c in enumerate(polar_cuts)
])

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    summary.to_excel(writer,  sheet_name="Summary",       index=False)
    gap_df.to_excel(writer,   sheet_name="Gap_Evolution", index=False)
    cuts_df.to_excel(writer,  sheet_name="Polar_Cuts",    index=False)
    extract_solution(df, m_lp_f, sm, dm).to_excel(writer, sheet_name="LP_final", index=False)
    extract_solution(df, m_ip_f, sm, dm).to_excel(writer, sheet_name="IP_final", index=False)
    extract_solution(df, m_lp0,  sm, dm).to_excel(writer, sheet_name="LP_before", index=False)
    extract_solution(df, m_ip0,  sm, dm).to_excel(writer, sheet_name="IP_before", index=False)

print(f"✅ Excel saved to: {output_file}")