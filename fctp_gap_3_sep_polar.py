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
input_file = "Input_data_3_sep.xlsx"
output_file = f"fctp_gap_3_sep_polar.xlsx"
log_file = "fctp_gap_3_Sep_polar_log.txt"       # <-- NEW
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
def build_model(df, cuts, polar_cuts, lp_relax=False):

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

    for pc in polar_cuts:
        expr = 0

        for i in pc["P"]:
            for j in pc["Q"]:
                expr += pc["pi_x"] * m.x[i, j]

        for i in pc["P"]:
            for j in pc["Q"]:
                expr += pc["pi_y"] * m.y[i, j]

        m.flowcover.add(expr <= 1)

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
# BUILD REDUCED SUBSTRUCTURE FROM LP_after SHEET
# ===============================================================

def build_substructure_from_excel(lp_df):

    lp_df = lp_df.copy()

    # Keep only positive flows
    lp_df = lp_df[lp_df["x_ij"] > EPS]

    sources = sorted(lp_df["Source"].unique())
    sinks = sorted(lp_df["Sink"].unique())

    P_names = sources
    Q_names = sinks

    P = []
    Q = []

    s = {}
    d = {}

    for src in P_names:
        idx = sm[src]
        P.append(idx)
        s[idx] = float(
        lp_df.loc[
        lp_df["Source"] == src,
        "a_i"
        ].iloc[0]
        )

    for dst in Q_names:
        idx = dm[dst]
        Q.append(idx)
        d[idx] = float(
        lp_df.loc[
        lp_df["Sink"] == dst,
        "b_j"
        ].iloc[0]
        )

    print("\nReduced Substructure")
    print("---------------------------")
    print("P =", P_names)
    print("Q =", Q_names)
    print("Supply =", s)
    print("Demand =", d)

    return P, Q, s, d

# ===============================================================
# SOLVE X(Y) SUBSTRUCTURE
# ===============================================================

def solve_substructure(P,Q,s,d):
    if len(P)==0 or len(Q)==0:
        return []

    A=[(i,j) for i in P for j in Q]

    M={(i,j):min(s[i],d[j]) for (i,j) in A}

    print("\nGenerating X(Y) Table")
    print("-------------------------")

    XY=[]

    for Y in range(len(A)+1):

        m=ConcreteModel()

        m.A=Set(initialize=A,dimen=2)

        m.P=Set(initialize=P)

        m.Q=Set(initialize=Q)

        m.x=Var(m.A,within=NonNegativeReals)

        m.y=Var(m.A,within=Binary)

        m.obj=Objective(

            expr=sum(m.x[a] for a in m.A),

            sense=maximize

        )

        m.sup=Constraint(

            m.P,

            rule=lambda m,i:

            sum(m.x[i,j] for j in m.Q)<=s[i]

        )

        m.dem=Constraint(

            m.Q,

            rule=lambda m,j:

            sum(m.x[i,j] for i in m.P)<=d[j]

        )

        m.link=Constraint(

            m.A,

            rule=lambda m,i,j:

            m.x[i,j]<=M[(i,j)]*m.y[i,j]

        )

        m.count=Constraint(

            expr=sum(m.y[a] for a in m.A)==Y

        )

        SolverFactory(solver_name).solve(m)

        X=value(m.obj)

        XY.append((Y,X))

        print(f"Y={Y:2d}   X={X:8.3f}")

    return XY

# ===============================================================
# SOLVE POLAR LP
# ===============================================================

def solve_polar_lp(lp_df, XY_table):

    total_x = lp_df["x_ij"].sum()
    total_y = lp_df["y_ij"].sum()

    print("\nCurrent LP Point")
    print("----------------")
    print(f"Total X = {total_x:.4f}")
    print(f"Total Y = {total_y:.4f}")

    m = ConcreteModel()

    m.pix = Var(within=Reals)
    m.piy = Var(within=Reals)

    m.obj = Objective(
        expr=total_x*m.pix + total_y*m.piy,
        sense=maximize
    )

    m.cuts = ConstraintList()

    for Y, X in XY_table:

        m.cuts.add(
            X*m.pix + Y*m.piy <= 1
        )

    SolverFactory(solver_name).solve(m)

    pix = value(m.pix)
    piy = value(m.piy)

    print("\nPolar LP")
    print("---------")
    print(f"pi_x = {pix:.6f}")
    print(f"pi_y = {piy:.6f}")

    return pix, piy

# ===============================================================
# BUILD POLAR CUT
# ===============================================================

def build_polar_cut(P, Q, pi_x, pi_y):

    print("\nGenerated Polar Valid Inequality")
    print("--------------------------------")

    lhs = []

    for i in P:
        for j in Q:
            lhs.append(f"{pi_x:.6f} x({i},{j})")

    for i in P:
        for j in Q:
            lhs.append(f"{pi_y:.6f} y({i},{j})")

    print(" + ".join(lhs))
    print("<= 1")


# ===============================================================
# MAIN CUTTING PLANE
# ===============================================================
df = pd.read_excel(input_file)
solver = SolverFactory(solver_name)

cuts = []
polar_cuts=[]
gap_log = []
iteration = 0

print("\n📘 FCTP Cutting Plane with Flow Cover Inequalities")
print(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------- BEFORE CUTS ----------
model_lp_b, sm, dm = build_model(
    df,
    cuts,
    polar_cuts,
    lp_relax=True
)

model_ip_b, _, _ = build_model(
    df,
    cuts,
    polar_cuts,
    lp_relax=False
)

t0 = time.time()
solver.solve(model_lp_b)
t_lp_b = time.time() - t0

t0 = time.time()
solver.solve(model_ip_b)
t_ip_b = time.time() - t0

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

    model_lp, _, _ = build_model(df,cuts,polar_cuts,lp_relax=True)
    solver.solve(model_lp)

    new_cuts = []
    new_cuts += separate_supply(model_lp)
    new_cuts += separate_demand(model_lp)

    model_ip_tmp, _, _ = build_model(df, cuts, polar_cuts, lp_relax=False)
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
model_lp_a, _, _ = build_model(df,cuts,polar_cuts,lp_relax=True)
model_ip_a, _, _ = build_model(df, cuts, polar_cuts, lp_relax=False)

t0 = time.time()
solver.solve(model_lp_a)
t_lp_a = time.time() - t0

t0 = time.time()
solver.solve(model_ip_a)
t_ip_a = time.time() - t0

lp_after = value(model_lp_a.obj)
ip_after = value(model_ip_a.obj)
gap_after = 100 * (lp_after - ip_after) / ip_after


    # ===============================================================
# START POLAR DUALITY PHASE
# ===============================================================

print("\n")
print("="*70)
print("STARTING POLAR DUALITY PHASE")
print("="*70)

lp_after_df = extract_solution(df,model_lp_a,sm,dm)

print("\nLatest LP_after Solution")
P,Q,s,d = build_substructure_from_excel(lp_after_df)
XY_table = []
XY_table=solve_substructure(P,Q,s,d)
if len(XY_table) > 0:
    pi_x, pi_y = solve_polar_lp(lp_after_df, XY_table)
else:
    raise RuntimeError("XY table is empty.")
build_polar_cut(
    P,
    Q,
    pi_x,
    pi_y
)

print("\nResolving LP after Polar Cut")

polar_iteration = 1

current_gap = gap_after

while True:

    # Rebuild ORIGINAL master
    model_lp, _, _ = build_model(
        df,
        cuts,
        polar_cuts,
        lp_relax=True
    )

    model_ip, _, _ = build_model(
        df,
        cuts,
        polar_cuts,
        lp_relax=False
    )

    solver.solve(model_lp)
    solver.solve(model_ip)

    lp_after_df = extract_solution(df,model_lp,sm,dm)
    P,Q,s,d = build_substructure_from_excel(lp_after_df)
    XY_table = solve_substructure(P,Q,s,d)
    pi_x,pi_y = solve_polar_lp(lp_after_df,XY_table)

    polar_cuts.append({

    "P":P.copy(),

    "Q":Q.copy(),

    "pi_x":pi_x,

    "pi_y":pi_y

    })

    lp_value = value(model_lp.obj)
    ip_value = value(model_ip.obj)

    new_gap = 100*(lp_value-ip_value)/ip_value

    # Write new LP_after

    # Read new LP_after

    # Build new P,Q,s,d

    # Compute new XY

    # Solve new Polar LP

    # Generate next Polar Cut

    print(
    f"\nPolar Iteration {polar_iteration}"
    )

    print(
    f"Gap = {new_gap:.4f}%"
    )

    if abs(current_gap-new_gap)<=EPS:
        print("\nNo further improvement.")
        break

    current_gap = new_gap
    build_polar_cut(P,Q,pi_x,pi_y)
    polar_iteration += 1

# ===============================================================
# SAVE EXCEL OUTPUTS (UNCHANGED)
# ===============================================================
with pd.ExcelWriter(output_file) as writer:

    summary = pd.DataFrame({
        "Metric": [
            "IP Objective (before)", "LP Objective (before)", "Gap% (before)",
            "IP Time (before)", "LP Time (before)",
            "IP Objective (after)", "LP Objective (after)", "Gap% (after)",
            "IP Time (after)", "LP Time (after)",
            "Gap Reduction%"
        ],
        "Value": [
            ip_before, lp_before, gap_before,
            round(t_ip_b, 3), round(t_lp_b, 3),
            ip_after, lp_after, gap_after,
            round(t_ip_a, 3), round(t_lp_a, 3),
            gap_before - gap_after
        ]
    })
    summary.to_excel(writer, "Summary", index=False)

    extract_solution(df, model_lp_b, sm, dm).to_excel(writer, "LP_before", index=False)
    extract_solution(df, model_ip_b, sm, dm).to_excel(writer, "IP_before", index=False)
    extract_solution(df, model_lp, sm, dm).to_excel(writer, "LP_after", index=False)
    extract_solution(df, model_ip_a, sm, dm).to_excel(writer, "IP_after", index=False)

# ===============================================================
# SAVE TERMINAL OUTPUT TO LOG FILE
# ===============================================================

sys.stdout = tee.terminal

with open(log_file, "w", encoding="utf-8") as f:
    f.write(tee.getvalue())

print(f"\n📄 Terminal log saved to: {log_file}")