
import pandas as pd  # Import Pandas to read Excel data.
from pyomo.environ import *  # Import Pyomo optimization functions.

# SECTION 1: INPUT PARAMETERS
P = 6  # Specify the exact number of FFCs to select.

# SECTION 2: READ EXCEL DATA
file = "Case Data.xlsx"  # Specify the Excel file name.
cities = pd.read_excel(file, sheet_name=0)  # Read the first worksheet.
names = cities.iloc[:, 1].astype(str).tolist()  # Extract city names.
demand = cities.iloc[:, 3].astype(float).tolist()  # Extract city demand.
n = len(names)  # Count the total number of cities.

distances = pd.read_excel(file, sheet_name=1, header=None)  # Read the distance matrix.
distances = distances.iloc[1:n+1, 1:n+1].to_numpy(dtype=float)  # Extract city distances.

# SECTION 3: CREATE MODEL AND DECISION VARIABLES
model = ConcreteModel()  # Create the optimization model.
model.y = Var(range(n), domain=Binary)  # y[j] = 1 if an FFC opens in city j.
model.x = Var(range(n), range(n), domain=Binary)  # x[i,j] = 1 if city i is assigned to FFC j.

# SECTION 4: SELECT EXACTLY P FFCS
model.facilities = Constraint(
    expr=sum(model.y[j] for j in range(n)) == P  # Require exactly P FFCs.
)

# SECTION 5: ASSIGN EVERY CITY TO EXACTLY ONE FFC
model.assignment = ConstraintList()  # Initialize assignment constraints.

for i in range(n):  # Process every city.
    model.assignment.add(
        sum(model.x[i, j] for j in range(n)) == 1
    )  # Assign each city to exactly one FFC.

# SECTION 6: ASSIGN CITIES ONLY TO OPEN FFCS
model.link = ConstraintList()  # Initialize facility-linking constraints.

for i in range(n):  # Process every city.
    for j in range(n):  # Check every potential FFC.
        model.link.add(
            model.x[i, j] <= model.y[j]
        )  # Prevent assignment to a closed FFC.

# SECTION 7: MINIMIZE TOTAL DEMAND-WEIGHTED DISTANCE
model.objective = Objective(
    expr=sum(
        demand[i] * distances[i, j] * model.x[i, j]
        for i in range(n) for j in range(n)
    ),  # Calculate total demand-weighted distance.
    sense=minimize  # Minimize the objective value.
)

# SECTION 8: SOLVE MODEL
solver = SolverFactory("cplex")  # Initialize the CPLEX solver.
results = solver.solve(model)  # Solve the optimization model.

# SECTION 9: DISPLAY SELECTED FFC CITIES
print("\nSelected FFC cities:")  # Print the results heading.

for j in range(n):  # Check every candidate city.
    if value(model.y[j]) > 0.5:  # Check whether an FFC is selected.
        print(names[j])  # Display the selected FFC city.

# SECTION 10: DISPLAY CITY ASSIGNMENTS
print("\nCity assignments:")  # Print the assignment heading.

for i in range(n):  # Check every city.
    for j in range(n):  # Check every potential serving FFC.
        if value(model.x[i, j]) > 0.5:  # Identify the assigned FFC.
            print(
                f"{names[i]} -> {names[j]} "
                f"({distances[i, j]:.2f} km)"
            )  # Print city, assigned FFC, and distance.

# SECTION 11: DISPLAY FINAL RESULTS
print("\nNumber of FFCs =", round(value(model.facilities.body)))  # Display the facility count.
print("Total demand-weighted distance =", value(model.objective))  # Display the objective value.
