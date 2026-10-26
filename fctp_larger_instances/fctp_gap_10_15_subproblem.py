from pyomo.environ import *

def solve_substructure(Y_value):
    """
    Solves the substructure LP for fixed Y
    corresponding exactly to the formulation in the paper
    """

    # -----------------------------
    # Sets from Fig. 3
    # -----------------------------
    P = [0, 2, 3, 4, 5, 6, 7, 8]           # supply nodes
    Q = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]       # demand nodes
    A = [(i, j) for i in P for j in Q]

    # -----------------------------
    # Data (from the figure)
    # ----------------------------
    s = {0: 4, 2: 3, 3: 5, 4: 7, 5: 5, 6: 6, 7: 8, 8: 8}          # supply
    d = {0: 2, 1: 2, 2: 3, 3: 6, 5: 1, 6: 7, 7: 2, 8: 1, 9: 2, 10: 2, 11: 5, 12: 2, 13: 7, 14: 5}    # demand

    M = {(i, j): min(s[i], d[j]) for (i, j) in A}

    # -----------------------------
    # Model
    # -----------------------------
    m = ConcreteModel()

    m.P = Set(initialize=P)
    m.Q = Set(initialize=Q)
    m.A = Set(initialize=A, dimen=2)

    # Variables
    m.x = Var(m.A, within=NonNegativeReals)
    m.y = Var(m.A, within=Binary)

    # -----------------------------
    # Objective: maximize X
    # X = sum_{(i,j) in A} x_ij
    # -----------------------------
    m.obj = Objective(
        expr=sum(m.x[i, j] for (i, j) in m.A),
        sense=maximize
    )

    # -----------------------------
    # Constraints
    # -----------------------------

    # Supply constraints
    def supply_rule(m, i):
        return sum(m.x[i, j] for j in Q) <= s[i]
    m.supply = Constraint(m.P, rule=supply_rule)

    # Demand constraints
    def demand_rule(m, j):
        return sum(m.x[i, j] for i in P) <= d[j]
    m.demand = Constraint(m.Q, rule=demand_rule)

    # Linking constraints
    def link_rule(m, i, j):
        return m.x[i, j] <= M[(i, j)] * m.y[i, j]
    m.link = Constraint(m.A, rule=link_rule)

    # Aggregation constraint: sum y_ij = Y
    m.aggregate_y = Constraint(
        expr=sum(m.y[i, j] for (i, j) in m.A) == Y_value
    )

    # -----------------------------
    # Solve
    # -----------------------------
    solver = SolverFactory("cplex")
    solver.solve(m, tee=False)

    X_max = value(m.obj)
    return X_max

print("Y   X_max")
print("----------")

for Y in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]:
    Xmax = solve_substructure(Y)
    print(f"{Y:<2}  {Xmax:.0f}")
