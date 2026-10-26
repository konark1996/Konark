# ================================================================
# FCTP-R CUTTING-PLANE ALGORITHM
# One Flow Cover Inequality (FCI) per iteration
# Supply + Demand FCI separation using CPLEX
#
# TIMING METRICS REPORTED (ONLY THREE TIMES):
# 1. Without cuts: Time = LP solving time + IP solving time
# 2. Separation time while searching violated cuts
# 3. Solving LP adding cuts
# ================================================================
import pandas as pd
import time
import sys
import io
from datetime import datetime
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Param,
    Var,
    NonNegativeReals,
    Binary,
    UnitInterval,
    Objective,
    Constraint,
    ConstraintList,
    minimize,
    maximize,
    value,
)
from pyomo.opt import SolverFactory, TerminationCondition
# ================================================================
# 1. SETTINGS
# ================================================================
input_file = "Input_data_3.xlsx"
output_file = "fctp_gap_3_3_one_cut.xlsx"
log_file = "fctp_gap_3_3_one_cut_log.txt"
solver_name = "cplex"
EPS = 1e-6
MAX_ITER = 1000
BASELINE_RUNS = 3
CPLEX_THREADS = 2
# ================================================================
# 2. PRINT TO TERMINAL AND LOG FILE
# ================================================================
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
# Store every cut that is added.
all_cuts = []
# ================================================================
# 3. CREATE CPLEX SOLVER
# ================================================================
def make_solver():
    solver = SolverFactory(solver_name)
    if solver_name.lower() == "cplex":
        solver.options["threads"] = CPLEX_THREADS
    return solver
# ================================================================
# 4. SOLVE MODEL
# ================================================================
def solve_model(solver, model, label):
    results = solver.solve(model, tee=False)
    status = results.solver.status
    termination = results.solver.termination_condition
    if termination not in (
        TerminationCondition.optimal,
        TerminationCondition.feasible,
    ):
        raise RuntimeError(
            f"{label} failed. "
            f"Status = {status}; "
            f"Termination = {termination}"
        )
    return results
# ================================================================
# 5. BUILD FCTP-R MODEL
# ================================================================
def build_model(df, cuts, lp_relax=True):
    # Source and sink node names.
    sources = df["Source"].unique().tolist()
    sinks = df["Sink"].unique().tolist()
    # Map names to numerical indices.
    smap = {s: i for i, s in enumerate(sources)}
    dmap = {d: j for j, d in enumerate(sinks)}
    # Supply and demand capacities.
    a = df.groupby("Source")["a_i"].first().to_dict()
    b = df.groupby("Sink")["b_j"].first().to_dict()
    # Fixed cost, variable cost and revenue.
    f = {}
    c = {}
    p = {}
    for _, row in df.iterrows():
        i = smap[row["Source"]]
        j = dmap[row["Sink"]]
        f[i, j] = float(row["FixedCost"])
        c[i, j] = float(row["VariableCost"])
        p[j] = float(row["profit_j"])
    # Linking constants.
    M = {
        (i, j): min(a[sources[i]], b[sinks[j]])
        for i in range(len(sources))
        for j in range(len(sinks))
    }
    # Create Pyomo model.
    m = ConcreteModel()
    m.I = RangeSet(0, len(sources) - 1)
    m.J = RangeSet(0, len(sinks) - 1)
    # Parameters.
    m.a = Param(
        m.I,
        initialize={i: a[sources[i]] for i in m.I}
    )
    m.b = Param(
        m.J,
        initialize={j: b[sinks[j]] for j in m.J}
    )
    m.M = Param(m.I, m.J, initialize=M)
    m.f = Param(m.I, m.J, initialize=f)
    m.c = Param(m.I, m.J, initialize=c)
    m.p = Param(m.J, initialize=p)
    # Transportation quantity.
    m.x = Var(
        m.I,
        m.J,
        within=NonNegativeReals
    )
    # Fixed-charge decision:
    # LP relaxation: 0 <= y <= 1.
    # Integer model: y is binary.
    m.y = Var(
        m.I,
        m.J,
        within=UnitInterval if lp_relax else Binary
    )
    # Maximize revenue minus transportation costs.
    m.obj = Objective(
        expr=sum(
            m.p[j] * m.x[i, j]
            - m.f[i, j] * m.y[i, j]
            - m.c[i, j] * m.x[i, j]
            for i in m.I
            for j in m.J
        ),
        sense=maximize
    )
    # Supply constraints.
    m.supply = Constraint(
        m.I,
        rule=lambda model, i:
        sum(model.x[i, j] for j in model.J)
        <= model.a[i]
    )
    # Demand constraints.
    m.demand = Constraint(
        m.J,
        rule=lambda model, j:
        sum(model.x[i, j] for i in model.I)
        <= model.b[j]
    )
    # Linking constraints.
    m.link = Constraint(
        m.I,
        m.J,
        rule=lambda model, i, j:
        model.x[i, j] <= model.M[i, j] * model.y[i, j]
    )
    # Add accumulated flow-cover inequalities.
    m.flowcover = ConstraintList()
    for cut in cuts:
        m.flowcover.add(cut["rule"](m))
    return m, smap, dmap
# ================================================================
# 6. GAP CALCULATION
# ================================================================
def gap_percent(lp_obj, ip_obj):
    if abs(ip_obj) < EPS:
        return float("nan")
    return 100.0 * (lp_obj - ip_obj) / abs(ip_obj)
# ================================================================
# 7. FIND FIRST VIOLATED SUPPLY FCI
# ================================================================
def find_one_violated_supply_cut(model_lp, solver):
    # Scan supply nodes in order.
    for i in model_lp.I:
        # Auxiliary knapsack model.
        kp = ConcreteModel()
        kp.J = list(model_lp.J)
        kp.z = Var(kp.J, within=Binary)
        # Minimize sum (1-y)*z.
        kp.obj = Objective(
            expr=sum(
                (1 - value(model_lp.y[i, j])) * kp.z[j]
                for j in kp.J
            ),
            sense=minimize
        )
        # Selected cover capacity must exceed supply.
        kp.cap = Constraint(
            expr=sum(
                value(model_lp.M[i, j]) * kp.z[j]
                for j in kp.J
            )
            >= value(model_lp.a[i]) + EPS
        )
        results = solver.solve(kp, tee=False)
        if (
            results.solver.termination_condition
            != TerminationCondition.optimal
        ):
            continue
        # Extract selected cover.
        cover = [
            j for j in kp.J
            if value(kp.z[j]) > 0.5
        ]
        if not cover:
            continue
        # Calculate lambda.
        lam = (
            sum(
                value(model_lp.M[i, j])
                for j in cover
            )
            - value(model_lp.a[i])
        )
        # Calculate the FCI left-hand side.
        lhs = (
            sum(
                value(model_lp.x[i, j])
                for j in cover
            )
            + sum(
                (
                    value(model_lp.M[i, j]) - lam
                )
                * (1 - value(model_lp.y[i, j]))
                for j in cover
            )
        )
        rhs = value(model_lp.a[i])
        # Return the first violated supply FCI.
        if lhs > rhs + EPS:
            coefficients = {
                j: value(model_lp.M[i, j]) - lam
                for j in cover
            }
            terms = [f"x({i},{j})" for j in cover]
            terms += [
                f"({coefficients[j]:.8f})*(1-y({i},{j}))"
                for j in cover
            ]
            inequality = (
                " + ".join(terms) + f" <= a({i})"
            )
            # Print complete cut information.
            print("\n" + "=" * 75)
            print("SUPPLY FLOW COVER INEQUALITY")
            print(f"Supply node index: {i}")
            print(f"Cover C1: {cover}")
            print(f"Lambda: {lam:.8f}")
            print(f"Inequality: {inequality}")
            print(f"LHS at current LP solution: {lhs:.8f}")
            print(f"RHS: {rhs:.8f}")
            print(f"Violation: {lhs - rhs:.8f}")
            print("=" * 75)
            # Define the cut for the main model.
            def cut_rule(m, i=i, cover=cover, lam=lam):
                return (
                    sum(m.x[i, j] for j in cover)
                    + sum(
                        (m.M[i, j] - lam) * (1 - m.y[i, j])
                        for j in cover
                    )
                    <= m.a[i]
                )
            return {
                "Cut_Type": "SUPPLY",
                "Node_Index": i,
                "Cover": str(cover),
                "Lambda": lam,
                "LHS": lhs,
                "RHS": rhs,
                "Violation": lhs - rhs,
                "Inequality": inequality,
                "rule": cut_rule
            }
    return None
# ================================================================
# 8. FIND FIRST VIOLATED DEMAND FCI
# ================================================================
def find_one_violated_demand_cut(model_lp, solver):
    # Scan demand nodes in order.
    for j in model_lp.J:
        # Auxiliary knapsack model.
        kp = ConcreteModel()
        kp.I = list(model_lp.I)
        kp.z = Var(kp.I, within=Binary)
        # Minimize sum (1-y)*z.
        kp.obj = Objective(
            expr=sum(
                (1 - value(model_lp.y[i, j])) * kp.z[i]
                for i in kp.I
            ),
            sense=minimize
        )
        # Selected cover capacity must exceed demand.
        kp.cap = Constraint(
            expr=sum(
                value(model_lp.M[i, j]) * kp.z[i]
                for i in kp.I
            )
            >= value(model_lp.b[j]) + EPS
        )
        results = solver.solve(kp, tee=False)
        if (
            results.solver.termination_condition
            != TerminationCondition.optimal
        ):
            continue
        # Extract selected cover.
        cover = [
            i for i in kp.I
            if value(kp.z[i]) > 0.5
        ]
        if not cover:
            continue
        # Calculate lambda.
        lam = (
            sum(
                value(model_lp.M[i, j])
                for i in cover
            )
            - value(model_lp.b[j])
        )
        # Calculate the FCI left-hand side.
        lhs = (
            sum(
                value(model_lp.x[i, j])
                for i in cover
            )
            + sum(
                (
                    value(model_lp.M[i, j]) - lam
                )
                * (1 - value(model_lp.y[i, j]))
                for i in cover
            )
        )
        rhs = value(model_lp.b[j])
        # Return the first violated demand FCI.
        if lhs > rhs + EPS:
            coefficients = {
                i: value(model_lp.M[i, j]) - lam
                for i in cover
            }
            terms = [f"x({i},{j})" for i in cover]
            terms += [
                f"({coefficients[i]:.8f})*(1-y({i},{j}))"
                for i in cover
            ]
            inequality = (
                " + ".join(terms) + f" <= b({j})"
            )
            # Print complete cut information.
            print("\n" + "=" * 75)
            print("DEMAND FLOW COVER INEQUALITY")
            print(f"Demand node index: {j}")
            print(f"Cover C1: {cover}")
            print(f"Lambda: {lam:.8f}")
            print(f"Inequality: {inequality}")
            print(f"LHS at current LP solution: {lhs:.8f}")
            print(f"RHS: {rhs:.8f}")
            print(f"Violation: {lhs - rhs:.8f}")
            print("=" * 75)
            # Define the cut for the main model.
            def cut_rule(m, j=j, cover=cover, lam=lam):
                return (
                    sum(m.x[i, j] for i in cover)
                    + sum(
                        (m.M[i, j] - lam) * (1 - m.y[i, j])
                        for i in cover
                    )
                    <= m.b[j]
                )
            return {
                "Cut_Type": "DEMAND",
                "Node_Index": j,
                "Cover": str(cover),
                "Lambda": lam,
                "LHS": lhs,
                "RHS": rhs,
                "Violation": lhs - rhs,
                "Inequality": inequality,
                "rule": cut_rule
            }
    return None
# ================================================================
# 9. EXTRACT SOLUTION FOR EXCEL
# ================================================================
def extract_solution(df, model, smap, dmap):
    out = df.copy()
    out["x_ij"] = out.apply(
        lambda row: value(
            model.x[
                smap[row["Source"]],
                dmap[row["Sink"]]
            ]
        ),
        axis=1
    )
    out["y_ij"] = out.apply(
        lambda row: value(
            model.y[
                smap[row["Source"]],
                dmap[row["Sink"]]
            ]
        ),
        axis=1
    )
    return out
# ================================================================
# 10. MAIN ALGORITHM
# ================================================================
try:
    total_start = time.perf_counter()
    # Read input Excel file.
    df = pd.read_excel(input_file)
    # Check required columns.
    required_columns = [
        "Source",
        "Sink",
        "a_i",
        "b_j",
        "FixedCost",
        "VariableCost",
        "profit_j"
    ]
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing input columns: {missing_columns}"
        )
    solver = make_solver()
    # Store accumulated cuts and results.
    cuts = []
    all_cuts = []
    baseline_results = []
    iteration_results = []
    # Timing counters for the cutting-plane process.
    separation_time = 0.0
    lp_withcuts_time = 0.0
    print("\n" + "=" * 75)
    print("FCTP-R FLOW COVER CUTTING-PLANE ALGORITHM")
    print("=" * 75)
    print(
        f"Timestamp: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Input file: {input_file}")
    print(f"Solver: {solver_name}")
    print(f"Baseline runs: {BASELINE_RUNS}")
    print(f"Maximum iterations: {MAX_ITER}")
    # ============================================================
    # 10A. BASELINE RUNS WITHOUT CUSTOM FCIs
    # ============================================================
    print("\n" + "=" * 75)
    print("BASELINE RUNS: WITHOUT CUSTOM FCI CUTS")
    print("=" * 75)
    for run in range(1, BASELINE_RUNS + 1):
        print(f"\n---------- BASELINE RUN {run} ----------")
        # Build LP and IP without custom cuts.
        model_lp, smap, dmap = build_model(
            df, [], lp_relax=True
        )
        model_ip, _, _ = build_model(
            df, [], lp_relax=False
        )
        # Solve baseline LP.
        t0 = time.perf_counter()
        solve_model(
            solver,
            model_lp,
            f"Baseline LP run {run}"
        )
        lp_time = time.perf_counter() - t0
        lp_obj = value(model_lp.obj)
        # Solve baseline IP.
        t0 = time.perf_counter()
        solve_model(
            solver,
            model_ip,
            f"Baseline IP run {run}"
        )
        ip_time = time.perf_counter() - t0
        ip_obj = value(model_ip.obj)
        gap = gap_percent(lp_obj, ip_obj)
        # Required baseline time: LP + IP.
        without_cuts_time = lp_time + ip_time
        baseline_results.append({
            "Run": run,
            "LP_Objective": lp_obj,
            "IP_Objective": ip_obj,
            "Gap_Percent": gap,
            "LP_Solve_Time_Seconds": lp_time,
            "IP_Solve_Time_Seconds": ip_time,
            "Without_Cuts_Total_Time_Seconds": without_cuts_time
        })
        print(f"LP objective: {lp_obj:.6f}")
        print(f"IP objective: {ip_obj:.6f}")
        print(f"Gap (%): {gap:.4f}")
        print(f"LP solve time: {lp_time:.4f} seconds")
        print(f"IP solve time: {ip_time:.4f} seconds")
        print(
            f"WITHOUT-CUTS TIME (LP + IP): "
            f"{without_cuts_time:.4f} seconds"
        )
        # Save first run's models for Excel solution sheets.
        if run == 1:
            model_lp_before = model_lp
            model_ip_before = model_ip
            sm = smap
            dm = dmap
    # Use baseline Run 1 for before/after gap comparison.
    lp_before = baseline_results[0]["LP_Objective"]
    ip_before = baseline_results[0]["IP_Objective"]
    gap_before = baseline_results[0]["Gap_Percent"]
    # ============================================================
    # 10B. INITIAL LP FOR CUTTING-PLANE PROCEDURE
    # ============================================================
    print("\n" + "=" * 75)
    print("STARTING ONE-CUT-AT-A-TIME SEPARATION")
    print("=" * 75)
    # Start with an empty cut list.
    model_lp_current, _, _ = build_model(
        df, cuts, lp_relax=True
    )
    # Time the initial LP solve separately.
    t0 = time.perf_counter()
    solve_model(
        solver,
        model_lp_current,
        "Initial LP with no custom cuts"
    )
    initial_lp_time = time.perf_counter() - t0
    lp_withcuts_time += initial_lp_time
    print(
        f"Initial LP objective: "
        f"{value(model_lp_current.obj):.6f}"
    )
    print(
        f"Initial LP solving time: "
        f"{initial_lp_time:.4f} seconds"
    )
    # ============================================================
    # 10C. SEPARATION AND ONE CUT PER ITERATION
    # ============================================================
    iteration = 0
    while iteration < MAX_ITER:
        print("\n" + "-" * 75)
        print(f"SEPARATION PASS {iteration + 1}")
        print("-" * 75)
        # Current LP solution is already available.
        current_lp_obj = value(model_lp_current.obj)
        print(f"Current LP objective: {current_lp_obj:.6f}")
        # Start separation timer.
        # This measures auxiliary knapsack solves and cut checking.
        # It does NOT measure the main LP solve or IP solves.
        t0 = time.perf_counter()
        # First scan supply nodes.
        new_cut = find_one_violated_supply_cut(
            model_lp_current,
            solver
        )
        # Only if no supply FCI is found, scan demand nodes.
        if new_cut is None:
            new_cut = find_one_violated_demand_cut(
                model_lp_current,
                solver
            )
        pass_separation_time = time.perf_counter() - t0
        separation_time += pass_separation_time
        print(
            f"Separation time for this pass: "
            f"{pass_separation_time:.4f} seconds"
        )
        # Stop when no violated candidate is found.
        if new_cut is None:
            print(
                "No violated supply or demand FCI found."
            )
            print("Separation procedure stopped.")
            break
        # Add exactly one cut.
        cuts.append(new_cut)
        iteration += 1
        cut_number = len(cuts)
        # Save complete cut information for Excel.
        all_cuts.append({
            "Cut_Number": cut_number,
            "Cut_Type": new_cut["Cut_Type"],
            "Node_Index": new_cut["Node_Index"],
            "Cover": new_cut["Cover"],
            "Lambda": new_cut["Lambda"],
            "LHS_Before_Adding": new_cut["LHS"],
            "RHS": new_cut["RHS"],
            "Violation": new_cut["Violation"],
            "Inequality": new_cut["Inequality"]
        })
        print(f"\nCUT {cut_number} ADDED")
        print(f"Type: {new_cut['Cut_Type']}")
        print(f"Node: {new_cut['Node_Index']}")
        print(f"Cover: {new_cut['Cover']}")
        print(f"Lambda: {new_cut['Lambda']:.8f}")
        print(
            f"Violation before adding: "
            f"{new_cut['Violation']:.8f}"
        )
        print(f"Total cuts added: {len(cuts)}")
        # --------------------------------------------------------
        # RE-SOLVE LP AFTER ADDING THE CUT
        # --------------------------------------------------------
        model_lp_current, _, _ = build_model(
            df, cuts, lp_relax=True
        )
        t0 = time.perf_counter()
        solve_model(
            solver,
            model_lp_current,
            f"LP after cut {cut_number}"
        )
        lp_time = time.perf_counter() - t0
        # Count main LP solves in LP solving time.
        lp_withcuts_time += lp_time
        lp_obj = value(model_lp_current.obj)
        print(
            f"LP objective after cut {cut_number}: "
            f"{lp_obj:.6f}"
        )
        print(
            f"LP solve time after cut {cut_number}: "
            f"{lp_time:.4f} seconds"
        )
        # --------------------------------------------------------
        # SOLVE IP FOR ITERATION-WISE GAP REPORTING
        # --------------------------------------------------------
        model_ip_iter, _, _ = build_model(
            df, cuts, lp_relax=False
        )
        solve_model(
            solver,
            model_ip_iter,
            f"IP after cut {cut_number}"
        )
        ip_obj = value(model_ip_iter.obj)
        gap = gap_percent(lp_obj, ip_obj)
        print(f"IP objective after cut {cut_number}: {ip_obj:.6f}")
        print(f"Gap after cut {cut_number} (%): {gap:.4f}")
        # Store this iteration's results.
        iteration_results.append({
            "Iteration": cut_number,
            "Cut_Type": new_cut["Cut_Type"],
            "Node_Index": new_cut["Node_Index"],
            "Cover": new_cut["Cover"],
            "Lambda": new_cut["Lambda"],
            "Violation_Before_Adding": new_cut["Violation"],
            "LP_Objective": lp_obj,
            "IP_Objective": ip_obj,
            "Gap_Percent": gap,
            "Separation_Time_This_Pass": pass_separation_time,
            "LP_Solve_Time_After_Cut": lp_time,
            "Cumulative_Separation_Time": separation_time,
            "Cumulative_LP_Time_With_Cuts": lp_withcuts_time
        })
        # Next pass starts again from the first supply node.
    if iteration >= MAX_ITER:
        print(
            f"Maximum iteration limit {MAX_ITER} reached."
        )
    # ============================================================
    # 10D. FINAL LP AND IP
    # ============================================================
    # The current LP model already includes all cuts and has been solved.
    model_lp_after = model_lp_current
    lp_after = value(model_lp_after.obj)
    # Solve final IP with all accumulated cuts.
    model_ip_after, _, _ = build_model(
        df, cuts, lp_relax=False
    )
    solve_model(
        solver,
        model_ip_after,
        "Final IP with all cuts"
    )
    ip_after = value(model_ip_after.obj)
    gap_after = gap_percent(lp_after, ip_after)
    # Required time WITH cuts:
    # Separation time + LP solving time adding cuts.
    total_time_with_cuts = (
        separation_time + lp_withcuts_time
    )
    # ============================================================
    # 10E. FINAL RESULTS AND TIMING COMPARISON
    # ============================================================
    print("\n" + "=" * 75)
    print("FINAL OPTIMIZATION RESULTS")
    print("=" * 75)
    print(f"Total cuts added: {len(cuts)}")
    print("\n--- BEFORE CUTS: BASELINE RUN 1 ---")
    print(f"LP objective: {lp_before:.6f}")
    print(f"IP objective: {ip_before:.6f}")
    print(f"Gap before cuts (%): {gap_before:.4f}")
    print("\n--- AFTER ALL CUTS ---")
    print(f"LP objective: {lp_after:.6f}")
    print(f"IP objective: {ip_after:.6f}")
    print(f"Gap after cuts (%): {gap_after:.4f}")
    print(
        f"Gap reduction (percentage points): "
        f"{gap_before - gap_after:.4f}"
    )
    # ------------------------------------------------------------
    # ONLY THREE TIMES REPORTED:
    # 1. Without cuts time (LP + IP)
    # 2. Separation time while searching violated cuts
    # 3. Solving LP adding cuts
    # ------------------------------------------------------------
    print("\n" + "=" * 75)
    print("TIMING RESULTS (THREE TIMES)")
    print("=" * 75)
    print("1. Without-cuts time (w/o cuts: LP + IP):")
    for result in baseline_results:
        print(
            f"   Run {result['Run']}: "
            f"LP = {result['LP_Solve_Time_Seconds']:.4f} s, "
            f"IP = {result['IP_Solve_Time_Seconds']:.4f} s, "
            f"Total w/o cuts = "
            f"{result['Without_Cuts_Total_Time_Seconds']:.4f} s"
        )
    print(
        f"\n2. Separation time while searching violated cuts: "
        f"{separation_time:.4f} seconds"
    )
    print(
        f"3. Solving LP adding cuts: "
        f"{lp_withcuts_time:.4f} seconds"
    )
    print(
        f"\n   Total time with cuts (Separation + Solving LP): "
        f"{total_time_with_cuts:.4f} seconds"
    )
    # ============================================================
    # 10F. PRINT ALL ADDED CUTS
    # ============================================================
    print("\n" + "=" * 75)
    print("ALL ADDED FLOW COVER INEQUALITIES")
    print("=" * 75)
    if not all_cuts:
        print("No violated FCI was added.")
    for cut in all_cuts:
        print(f"\nCut number: {cut['Cut_Number']}")
        print(f"Type: {cut['Cut_Type']}")
        print(f"Node index: {cut['Node_Index']}")
        print(f"Cover: {cut['Cover']}")
        print(f"Lambda: {cut['Lambda']:.8f}")
        print(f"Inequality: {cut['Inequality']}")
        print(
            f"LHS before adding: "
            f"{cut['LHS_Before_Adding']:.8f}"
        )
        print(f"RHS: {cut['RHS']:.8f}")
        print(f"Violation: {cut['Violation']:.8f}")
    # ============================================================
    # 10G. CREATE EXCEL OUTPUT
    # ============================================================
    baseline_df = pd.DataFrame(baseline_results)
    cuts_df = pd.DataFrame(all_cuts)
    iterations_df = pd.DataFrame(iteration_results)
    # Summary includes only the three requested times:
    summary = pd.DataFrame({
        "Metric": [
            "LP Objective Before Cuts (Baseline Run 1)",
            "IP Objective Before Cuts (Baseline Run 1)",
            "Gap Before Cuts (%)",
            "LP Objective After Cuts",
            "IP Objective After Cuts",
            "Gap After Cuts (%)",
            "Gap Reduction (percentage points)",
            "Number of FCIs Added",
            "1. Time Without Cuts: LP + IP (Baseline Run 1) (seconds)",
            "2. Separation Time While Searching Violated Cuts (seconds)",
            "3. Solving LP Adding Cuts (seconds)",
            "Total Time With Cuts: Separation + Solving LP (seconds)"
        ],
        "Value": [
            lp_before,
            ip_before,
            gap_before,
            lp_after,
            ip_after,
            gap_after,
            gap_before - gap_after,
            len(cuts),
            baseline_results[0]["Without_Cuts_Total_Time_Seconds"],
            separation_time,
            lp_withcuts_time,
            total_time_with_cuts
        ]
    })
    # Save all results and solutions.
    with pd.ExcelWriter(output_file) as writer:
        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )
        baseline_df.to_excel(
            writer,
            sheet_name="Baseline_Runs",
            index=False
        )
        cuts_df.to_excel(
            writer,
            sheet_name="All_Cuts",
            index=False
        )
        iterations_df.to_excel(
            writer,
            sheet_name="Iterations",
            index=False
        )
        extract_solution(
            df, model_lp_before, sm, dm
        ).to_excel(
            writer,
            sheet_name="LP_before",
            index=False
        )
        extract_solution(
            df, model_ip_before, sm, dm
        ).to_excel(
            writer,
            sheet_name="IP_before",
            index=False
        )
        extract_solution(
            df, model_lp_after, sm, dm
        ).to_excel(
            writer,
            sheet_name="LP_after",
            index=False
        )
        extract_solution(
            df, model_ip_after, sm, dm
        ).to_excel(
            writer,
            sheet_name="IP_after",
            index=False
        )
    print(f"\nExcel results saved to: {output_file}")
except Exception as exc:
    print(f"\nERROR: {exc}")
    print(
        "Check the input file, required columns, "
        "CPLEX installation, and solver status."
    )
    raise
finally:
    # Restore normal terminal output.
    sys.stdout = tee.terminal
    # Save all captured output to the log.
    with open(log_file, "w", encoding="utf-8") as log:
        log.write(tee.getvalue())
    print(f"\nComplete log saved to: {log_file}")
