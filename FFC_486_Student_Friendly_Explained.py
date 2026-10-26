# Input: Case Data.xlsx (city data in sheet 1, distances in sheet 2).
# Output: FFC_486_Objective_Results.xlsx (best cases and final comparison).
# Model: Open exactly P facilities and maximize total demand covered.


# -------------------- 0. Import required libraries --------------------

import pandas as pd  # Read Excel files and create result tables.
from pyomo.environ import (  # Import tools for building and solving optimization models.
    ConcreteModel, Var, Binary, Constraint, Objective,  # Model, variables, binary values, constraints, objective.
    SolverFactory, maximize, quicksum, value  # Solver connection, maximize direction, sums and value extraction.
)


# -------------------- 1. Basic settings --------------------

INPUT_FILE = "Case Data.xlsx"  # Name of the input workbook.
OUTPUT_FILE = "FFC_486_Objective_Results.xlsx"  # Name of the output workbook.

P_VALUES = [8, 10, 15]  # Test cases with 8, 10 and 15 facilities.
RADII = {"SDD": 200, "NDD": 400}  # Delivery radius in kilometres for each category.
SOLVER_NAME = "cplex"  # Solver to use; change this if another installed solver is preferred.
TIME_LIMIT = 300  # Maximum solver time in seconds, where supported.

# -------------------- 2. Read the Excel data --------------------

cities = pd.read_excel(INPUT_FILE, sheet_name=0).iloc[:, :4].copy()  # Read the first four columns from sheet 1.
cities.columns = ["id", "name", "tier", "demand"]  # Assign clear names to the four columns.
cities = cities.dropna(subset=["name", "tier", "demand"]).reset_index(drop=True)  # Remove rows missing key data and reset row numbers.
cities["demand"] = pd.to_numeric(cities["demand"], errors="raise")  # Ensure demand values are numbers.

distance_sheet = pd.read_excel(INPUT_FILE, sheet_name=1, header=None)  # Read sheet 2 without treating its first row as headings.
n = len(cities)  # Count the valid city records.
distances = distance_sheet.iloc[1:n + 1, 1:n + 1].apply(  # Skip the first row and column containing labels.
    pd.to_numeric, errors="raise"  # Convert all selected distance entries to numbers.
).to_numpy(dtype=float)  # Store the distance matrix as a numerical array.

if distances.shape != (n, n):  # Check that the matrix has one row and column per city.
    raise ValueError(f"Expected a {n} x {n} distance matrix; found {distances.shape}.")  # Stop if its dimensions are wrong.

if n != 486:  # Confirm that the expected 486 cities are present.
    raise ValueError(f"Expected 486 cities; found {n}.")  # Stop if the city count differs.

if not (distances >= 0).all():  # Check that no distance is negative.
    raise ValueError("Distances must be non-negative.")  # Stop if invalid distances are found.

demand = cities["demand"].to_numpy(dtype=float)  # Store city demands in a numerical array.

# -------------------- 3. Find which FFCs can cover each city --------------------

neighbourhoods = {}  # Store eligible FFC locations for every city and radius.

for delivery_name, radius in RADII.items():  # Process SDD and NDD one at a time.
    neighbourhoods[delivery_name] = []  # Start a list of coverage sets for this radius.
    for i in range(n):  # Check each city as a demand city.
        possible_sites = [  # Collect candidate FFC sites close enough to cover city i.
            j for j in range(n) if distances[i, j] <= radius  # Include site j if its distance is within the radius.
        ]
        neighbourhoods[delivery_name].append(possible_sites)  # Save the eligible sites for city i.

# -------------------- 4. Solve the six optimization cases --------------------

best_results = []  # Store the result for each facility-count and radius combination.

for delivery_name, radius in RADII.items():  # Run once for each delivery radius.
    for p in P_VALUES:  # Test each requested number of facilities.
        print(f"\nSolving: P={p}, {delivery_name}, radius={radius} km")  # Display the case being solved.

        model = ConcreteModel()  # Create a new optimization model.

        model.y = Var(range(n), domain=Binary)  # y[j] = 1 if an FFC opens at site j; otherwise 0.
        model.x = Var(range(n), domain=Binary)  # x[i] = 1 if demand city i is covered; otherwise 0.

        model.coverage = Constraint(  # Add one coverage constraint for each demand city.
            range(n),  # Use all city indices to create the constraints.
            rule=lambda m, i: m.x[i] <= quicksum(  # A city can be covered only if at least one eligible FFC is open.
                m.y[j] for j in neighbourhoods[delivery_name][i]  # Sum facility variables for sites that can cover city i.
            )
        )

        model.number_of_ffcs = Constraint(  # Add the facility-count constraint.
            expr=quicksum(model.y[j] for j in range(n)) == p  # Require exactly p facilities to open.
        )

        model.objective = Objective(  # Define the objective function.
            expr=quicksum(demand[i] * model.x[i] for i in range(n)),  # Add demand only for cities marked as covered.
            sense=maximize  # Maximize total covered demand.
        )

        solver = SolverFactory(SOLVER_NAME)  # Create an interface to the selected solver.
        if not solver.available(exception_flag=False):  # Check whether the solver is accessible.
            raise RuntimeError(  # Stop with a clear message if the solver is unavailable.
                f"Solver '{SOLVER_NAME}' is not available. "  # Identify the unavailable solver.
                "Install/configure it or change SOLVER_NAME."  # Explain how to resolve the issue.
            )

        if SOLVER_NAME == "cplex":  # Apply the time limit using CPLEX's option name.
            solver.options["timelimit"] = TIME_LIMIT  # Set the maximum CPLEX solve time.
        elif SOLVER_NAME == "gurobi":  # Apply the time limit for Gurobi.
            solver.options["TimeLimit"] = TIME_LIMIT  # Set the maximum Gurobi solve time.
        elif SOLVER_NAME == "scip":  # Apply the time limit for SCIP.
            solver.options["limits/time"] = TIME_LIMIT  # Set the maximum SCIP solve time.
        elif SOLVER_NAME == "cbc":  # Apply the time limit for CBC.
            solver.options["seconds"] = TIME_LIMIT  # Set the maximum CBC solve time.
        elif SOLVER_NAME == "highs":  # Apply the time limit for HiGHS.
            solver.options["time_limit"] = TIME_LIMIT  # Set the maximum HiGHS solve time.

        result = solver.solve(model, tee=False)  # Solve the model without displaying the solver's detailed log.
        status = str(result.solver.termination_condition)  # Record how the solver finished.

        if status not in ("optimal", "feasible", "maxTimeLimit", "maxTime"):  # Check whether a usable solution may exist.
            print(f"  No usable solution. Solver status: {status}")  # Explain why this case was not solved.
            best_results.append({  # Keep a record so this case is not silently omitted.
                "P": p,  # Save the number of facilities.
                "Delivery": delivery_name,  # Save the delivery category.
                "Radius_km": radius,  # Save the radius in kilometres.
                "Objective_value": None,  # Record that no objective value is available.
                "Status": status,  # Save the solver status.
                "FFC_Cities": ""  # Leave selected city names blank.
            })
            continue  # Move to the next case.

        objective_value = value(model.objective, exception=False)  # Retrieve total covered demand.
        if objective_value is None:  # Check that the objective could be evaluated.
            print(f"  Solver returned no objective. Status: {status}")  # Report the missing objective.
            best_results.append({  # Save a record for this unsuccessful case.
                "P": p,  # Save the number of facilities.
                "Delivery": delivery_name,  # Save the delivery category.
                "Radius_km": radius,  # Save the radius.
                "Objective_value": None,  # Mark the objective as unavailable.
                "Status": status,  # Save the solver status.
                "FFC_Cities": ""  # No selected city names are recorded.
            })
            continue  # Move to the next case.

        selected_indices = [  # Find the candidate locations selected by the solver.
            j for j in range(n) if value(model.y[j]) > 0.5  # A binary variable above 0.5 is treated as selected.
        ]
        selected_city_names = [  # Convert selected location indices to city names.
            str(cities.iloc[j]["name"]) for j in selected_indices  # Read the name from the matching city row.
        ]

        print(f"  Covered demand: {objective_value:.4f}")  # Display the objective value.
        print(f"  Selected FFC cities: {selected_city_names}")  # Display the selected facility cities.
        print(f"  Solver status: {status}")  # Display whether the solver proved optimality or stopped another way.

        best_results.append({  # Save this case's results.
            "P": p,  # Record the number of facilities.
            "Delivery": delivery_name,  # Record SDD or NDD.
            "Radius_km": radius,  # Record the delivery radius.
            "Objective_value": float(objective_value),  # Record total covered demand as a number.
            "Status": status,  # Record the solver termination status.
            "FFC_Cities": ", ".join(selected_city_names)  # Store selected city names as one comma-separated string.
        })

# -------------------- 5. Create the final comparison table --------------------

best_by_case = pd.DataFrame(best_results)  # Convert all case results into a table.

final_matrix = best_by_case.pivot(  # Arrange results into one row per facility count.
    index="P", columns="Delivery", values="Objective_value"  # Put SDD and NDD objective values in separate columns.
).reindex(P_VALUES)  # Display the rows in the order P = 8, 10, 15.

final_matrix = final_matrix.rename(columns={  # Give the delivery columns clear names.
    "SDD": "SDD objective (200 km)",  # Label the 200 km objective column.
    "NDD": "NDD objective (400 km)"  # Label the 400 km objective column.
}).reset_index().rename(columns={"P": "Number of FFCs (P)"})  # Make P a regular column with a descriptive heading.

print("\nFINAL MATRIX — TOTAL DEMAND COVERED")  # Print a heading for the final comparison.
print(final_matrix.to_string(index=False, float_format=lambda x: f"{x:.4f}"))  # Print values to four decimal places.

# -------------------- 6. Save the results --------------------

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:  # Open an Excel workbook for the output.
    best_by_case.to_excel(writer, sheet_name="Best by Case", index=False)  # Save city selections and case details.
    final_matrix.to_excel(writer, sheet_name="Final Matrix", index=False)  # Save the compact objective comparison.

print(f"\nResults saved to: {OUTPUT_FILE}")  # Display the saved workbook's filename.
