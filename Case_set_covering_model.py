import pandas as pd  # Import Pandas to read Excel data.
from pyomo.environ import *  # Import Pyomo optimization functions.

# SECTION 1: INPUT PARAMETERS
RADIUS = 400  # Maximum distance for a city to be covered (km).

# SECTION 2: READ EXCEL DATA
file = "Case Data.xlsx"  # Specify the Excel file name.
cities = pd.read_excel(file, sheet_name=0)  # Read the first worksheet.
names = cities.iloc[:, 1].astype(str).tolist()  # Extract city names.
n = len(names)  # Count the number of cities.

distances = pd.read_excel(file, sheet_name=1, header=None)  # Read the distance matrix.
distances = distances.iloc[1:n+1, 1:n+1].to_numpy(dtype=float)  # Extract city distances.

# SECTION 3: CREATE MODEL
model = ConcreteModel()  # Create the optimization model.
model.y = Var(range(n), domain=Binary)  # y[j] = 1 if an FFC opens in city j.

# SECTION 4: MINIMIZE NUMBER OF FFCS
model.objective = Objective(
    expr=sum(model.y[j] for j in range(n)),  # Count selected FFCs.
    sense=minimize  # Minimize the total number of FFCs.
)

# SECTION 5: COVER EVERY CITY
model.coverage = ConstraintList()  # Initialize coverage constraints.

for i in range(n):  # Process every city.
    model.coverage.add(
        sum(model.y[j] for j in range(n)
            if distances[i, j] <= RADIUS) >= 1
    )  # Require at least one FFC within the radius of city i.

# SECTION 6: SOLVE MODEL
solver = SolverFactory("cplex")  # Initialize the CPLEX solver.
results = solver.solve(model)  # Solve the optimization model.

# SECTION 7: DISPLAY RESULTS
print("\nMinimum number of FFCs =", round(value(model.objective)))  # Print minimum FFC count.
print("\nSelected FFC cities:")  # Print the results heading.

for j in range(n):  # Check every candidate city.
    if value(model.y[j]) > 0.5:  # Check whether an FFC is selected.
        print(names[j])  # Display the selected city name.