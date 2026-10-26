# polar_step1_FINAL_correct.py

import pandas as pd
from pyomo.environ import *

# -------------------------------
# SETTINGS
# -------------------------------
input_file = "FCTP_gap_1.xlsx"
sheet_name = "LP_after"
solver_name = "cplex"
# -------------------------------


# =========================================================
# LOAD DATA
# =========================================================
print(f"\n📘 Reading: {input_file} ({sheet_name})")

df_full = pd.read_excel(input_file, sheet_name=sheet_name)

# -------------------------------
# STEP 1: FILTER ACTIVE ARCS
# -------------------------------
df_active = df_full[df_full["x_ij"] > 1e-6].copy()

print(f"✅ Active arcs: {len(df_active)}")

# -------------------------------
# STEP 2: BUILD P and Q (ONLY ACTIVE NODES)
# -------------------------------
sources = df_active["Source"].unique().tolist()
sinks = df_active["Sink"].unique().tolist()

source_map = {s: i for i, s in enumerate(sources)}
sink_map = {t: j for j, t in enumerate(sinks)}

P = list(range(len(sources)))
Q = list(range(len(sinks)))

# -------------------------------
# STEP 3: SUPPLY & DEMAND (from FULL DATA)
# -------------------------------
a_i_series = df_full.groupby("Source")["a_i"].first()
b_j_series = df_full.groupby("Sink")["b_j"].first()

s = {source_map[k]: float(a_i_series[k]) for k in sources}
d = {sink_map[k]: float(b_j_series[k]) for k in sinks}

# -------------------------------
# STEP 4: FULL ARC SET (CRITICAL FIX 🔥)
# -------------------------------
A = [(i, j) for i in P for j in Q]

# -------------------------------
# STEP 5: Big-M
# -------------------------------
M = {(i, j): min(s[i], d[j]) for (i, j) in A}


# =========================================================
# SUBPROBLEM (UNCHANGED STRUCTURE)
# =========================================================
def solve_substructure(Y_value):

    m = ConcreteModel()

    m.P = Set(initialize=P)
    m.Q = Set(initialize=Q)
    m.A = Set(initialize=A, dimen=2)

    # Variables
    m.x = Var(m.A, within=NonNegativeReals)
    m.y = Var(m.A, within=Binary)

    # Objective
    m.obj = Objective(
        expr=sum(m.x[i, j] for (i, j) in m.A),
        sense=maximize
    )

    # Supply
    def supply_rule(m, i):
        return sum(m.x[i, j] for j in Q) <= s[i]
    m.supply = Constraint(m.P, rule=supply_rule)

    # Demand
    def demand_rule(m, j):
        return sum(m.x[i, j] for i in P) <= d[j]
    m.demand = Constraint(m.Q, rule=demand_rule)

    # Linking
    def link_rule(m, i, j):
        return m.x[i, j] <= M[(i, j)] * m.y[i, j]
    m.link = Constraint(m.A, rule=link_rule)

    # Polar constraint
    m.aggregate_y = Constraint(
        expr=sum(m.y[i, j] for (i, j) in m.A) == Y_value
    )

    # Solve
    solver = SolverFactory(solver_name)
    result = solver.solve(m, tee=False)

    if (result.solver.status != SolverStatus.ok) or \
       (result.solver.termination_condition != TerminationCondition.optimal):
        return None

    return value(m.obj)


# =========================================================
# MAIN LOOP
# =========================================================
print("\n📊 STEP-1: Polar Duality (Y vs X_max)")
print("Y   X_max")
print("-----------")

max_possible_Y = len(A)

results = []

for Y in range(0, max_possible_Y + 1):

    Xmax = solve_substructure(Y)

    if Xmax is None:
        print(f"{Y:<3} infeasible")
    else:
        print(f"{Y:<3} {Xmax:.2f}")
        results.append((Y, Xmax))


# =========================================================
# SAVE
# =========================================================
df_results = pd.DataFrame(results, columns=["Y", "X_max"])
df_results.to_excel("Polar_Step1_FINAL.xlsx", index=False)

print("\n✅ FINAL correct results saved.")