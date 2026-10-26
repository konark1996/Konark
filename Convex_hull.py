pip install docplex
pip install cplex

from docplex.mp.model import Model

# Create model
mdl = Model(name="Polar_Dual")

# -----------------------------
# Decision Variables (Unrestricted)
# -----------------------------
x00 = mdl.continuous_var(lb=-mdl.infinity, name='x00')
x01 = mdl.continuous_var(lb=-mdl.infinity, name='x01')
x02 = mdl.continuous_var(lb=-mdl.infinity, name='x02')
x03 = mdl.continuous_var(lb=-mdl.infinity, name='x03')

y00 = mdl.continuous_var(lb=-mdl.infinity, name='y00')
y01 = mdl.continuous_var(lb=-mdl.infinity, name='y01')
y02 = mdl.continuous_var(lb=-mdl.infinity, name='y02')
y03 = mdl.continuous_var(lb=-mdl.infinity, name='y03')

# -----------------------------
# Objective Function
# -----------------------------
mdl.maximize(
    12 * (x00 + x01 + x02 + x03)
    + 3.17 * (y00 + y01 + y02 + y03)
)

# -----------------------------
# Constraints
# -----------------------------
mdl.add_constraint(4*x00 + x01 + y00 + y01 <= 1)
mdl.add_constraint(4*x00 + x02 + y00 + y02 <= 1)
mdl.add_constraint(4*x00 + x03 + y00 + y03 <= 1)
mdl.add_constraint(x00 + 4*x01 + y00 + y01 <= 1)
mdl.add_constraint(x00 + 4*x02 + y00 + y02 <= 1)
mdl.add_constraint(x00 + 4*x03 + y00 + y03 <= 1)
mdl.add_constraint(4*x01 + x02 + y01 + y02 <= 1)
mdl.add_constraint(4*x01 + x03 + y01 + y03 <= 1)
mdl.add_constraint(x01 + 4*x02 + y01 + y02 <= 1)
mdl.add_constraint(x01 + 4*x03 + y01 + y03 <= 1)
mdl.add_constraint(4*x02 + x03 + y02 + y03 <= 1)
mdl.add_constraint(x02 + 4*x03 + y02 + y03 <= 1)

# -----------------------------
# Solve
# -----------------------------
solution = mdl.solve(log_output=True)

# -----------------------------
# Print Results
# -----------------------------
if solution:
    print("\n==============================")
    print("OPTIMAL SOLUTION FOUND")
    print("==============================")
    print(f"Objective Value = {mdl.objective_value:.6f}\n")

    for var in mdl.iter_variables():
        print(f"{var.name:5s} = {var.solution_value:.6f}")

    print("\nConstraint Slacks:")
    for c in mdl.iter_constraints():
        print(f"{c}: Slack = {c.slack_value:.6f}")

else:
    print("No feasible solution found.")