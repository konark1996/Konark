import pandas as pd  # Import Pandas for Excel data handling.
from pyomo.environ import *  # Import Pyomo optimization components.

# SECTION 1: SET INPUT PARAMETERS

P = 8  # Specify the exact number of FFCs to select.
RADIUS = 200  # Specify the maximum coverage radius in km.

# SECTION 2: READ AND PREPARE EXCEL DATA

file = "Case Data.xlsx"  # Specify the Excel file name.
cities = pd.read_excel(file, sheet_name=0)  # Read the first worksheet.
names = cities.iloc[:, 1].astype(str).tolist()  # Extract city names from column 2.
demand = cities.iloc[:, 3].astype(float).tolist()  # Extract demand from column 4.

distances = pd.read_excel(  # Start reading the distance matrix.
    file, sheet_name=1, header=None  # Read the second worksheet without headers.
).iloc[1:len(names)+1, 1:len(names)+1].to_numpy(dtype=float)  # Extract the city distance matrix.

n = len(names)  # Count the total number of cities.

# SECTION 3: CREATE MODEL AND DECISION VARIABLES

model = ConcreteModel()  # Initialize the Pyomo optimization model.
model.y = Var(range(n), domain=Binary)  # y[j] = 1 if an FFC is selected at city j.
model.x = Var(range(n), domain=Binary)  # x[i] = 1 if city i is covered.

# SECTION 4: SELECT EXACTLY P FFCS Constraint 1

model.facilities = Constraint(  # Define the facility-count constraint.
    expr=sum(model.y[j] for j in range(n)) == P  # Require exactly P selected FFCs.
)

# SECTION 5: ADD CITY COVERAGE CONSTRAINT 2

model.coverage = ConstraintList()  # Initialize the coverage constraint list.

for i in range(n):  # Process every city whose demand may be covered.
    nearby = []  # Store candidate FFCs within the coverage radius.

    for j in range(n):  # Check every potential FFC location.
        if distances[i, j] <= RADIUS:  # Check whether city j is within the radius of city i.
            nearby.append(model.y[j])  # Add the nearby FFC selection variable.

    model.coverage.add(  # Add the coverage constraint for city i.
        model.x[i] <= sum(nearby)  # Permit coverage only if a nearby FFC is selected.
    )

# SECTION 6: MAXIMIZE TOTAL COVERED DEMAND

model.objective = Objective(  # Define the optimization objective.
    expr=sum(demand[i] * model.x[i] for i in range(n)),  # Sum demand of covered cities.
    sense=maximize  # Maximize total covered demand.
)

# SECTION 7: SOLVE THE OPTIMIZATION MODEL

solver = SolverFactory("cplex")  # Initialize the CPLEX optimization solver.
results = solver.solve(model, tee=False)  # Solve the model and store solver results.


# SECTION 8: DISPLAY SELECTED FFC CITIES

print("\nP =", P, "| Radius =", RADIUS, "km")  # Display the facility count and radius.
print("\nSelected FFC cities:")  # Print the selected facility heading.

for j in range(n):  # Check every candidate FFC location.
    if value(model.y[j]) > 0.5:  # Check whether an FFC was selected.
        print(names[j])  # Print the selected FFC city name.

# SECTION 9: DISPLAY COVERED CITIES

print("\nCovered cities:")  # Print the covered-city heading.
count = 0  # Initialize the covered-city counter.

for i in range(n):  # Check every city.
    if value(model.x[i]) > 0.5:  # Check whether the city is marked as covered.
        print(names[i])  # Print the covered city name.
        count += 1  # Increment the covered-city count.

# SECTION 10: DISPLAY FINAL PERFORMANCE METRICS

print("\nNumber of cities covered =", count)  # Display the number of covered cities.
print("Total demand covered =", value(model.objective))  # Display the maximum covered demand.
