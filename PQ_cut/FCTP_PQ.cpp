// ============================================================================
// Single-stage FCTP with first-order PQ inequality 
// ============================================================================

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iosfwd>
#include <string>
#include <deque>
#include <sstream>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <chrono>

#include <ilcplex/ilocplex.h>
#include <ilconcert/ilosys.h>

using namespace std;

ILOSTLBEGIN;
typedef IloArray<IloNumArray> Num2DMatrix;
typedef IloArray<IloBoolVarArray> NumBoolVar2DMatrix; 
typedef IloArray<IloNumVarArray> array2d;

// -----------------------------------------------------------------------------
// solveKnapsackAndAddPQcuts
//
// Solves the knapsack separation problem and iteratively adds PQ cuts
// until no further cuts are found.
//
// Parameters:
//   @param env     CPLEX environment
//   @param model   The original MIP model to which PQ cuts are added
//   @param cplex1  The CPLEX object for solving the LP relaxation
//   @param x       2D matrix of flow variables (continuous)
//   @param y       2D matrix of binary open/close variables
//   @param xbar    2D matrix to store flow solution values
//   @param ybar    2D matrix to store binary solution values (fractional in LP)
//   @param u       Supplier indicator variables for knapsack
//   @param v       Customer indicator variables for knapsack
//   @param S       Supply capacities
//   @param D       Demand requirements
//   @param TD      Total supply (scalar)
//   @param P       Number of suppliers
//   @param Q       Number of customers
//   @param pqcount Reference counter for number of PQ cuts added
// -----------------------------------------------------------------------------
void solveKnapsackAndAddPQcuts(
    IloEnv &env,
    IloModel &model,
    IloCplex &cplex1,
    array2d &x,
    NumBoolVar2DMatrix &y,
    IloNumArray2 &xbar,
    IloNumArray2 &ybar,
    IloBoolVarArray &u,
    IloBoolVarArray &v,
    IloNumArray &S,
    IloNumArray &D,
    const IloNum TD,
    const IloInt P,
    const IloInt Q,
    int &pqcount
);

// -----------------------------------------------------------------------------
// AddPQCut
//
// Adds a single PQ cut to the master model.
//
// Parameters:
//   @param env     CPLEX environment
//   @param model   The original MIP model to which the PQ cut is added
//   @param y       2D matrix of binary open/close variables
//   @param uval    Solution values for knapsack u variables
//   @param vval    Solution values for knapsack v variables
//   @param P       Number of suppliers
//   @param Q       Number of customers
// -----------------------------------------------------------------------------
void AddPQCut(
    IloEnv &env,
    IloModel &model,
    const NumBoolVar2DMatrix &y,
    const IloNumArray &uval,
    const IloNumArray &vval,
    const IloInt P,
    const IloInt Q
);

int main(int argc, char **argv) {

    clock_t t1, t2;
    t1 = clock();

    // Input filenames for instances
    const char* filename[] = { "1.csv", "2.csv" };

    for (int k = 0; k < 1; k++) {
        IloEnv env;
        Num2DMatrix cp(env), fp(env);
        IloNumArray S(env), D(env);
        IloNum P, Q, TD = 0;

        // ----------------------------------------------------------
        // Read data from CSV file
        // ----------------------------------------------------------
        ifstream fin;
        fin.open(filename[k]);
        fin >> P >> Q >> S >> D >> fp >> cp;  // Assuming operator>> is overloaded for these types
        fin.close();

        P = S.getSize();
        Q = fp[0].getSize();

        try {
            IloModel model(env);

            // Variable declarations
            array2d x(env, P);                 // Flow variables
            NumBoolVar2DMatrix y(env, P);      // Binary open/close variables
            IloNumArray2 ybar(env, P), xbar(env, P);  // To store solution values
            IloBoolVarArray u(env, P), v(env, Q);     // Knapsack separation variables

            for (IloInt i = 0; i < P; i++) {
                x[i] = IloNumVarArray(env, Q, 0, IloInfinity, ILOFLOAT);
                y[i] = IloBoolVarArray(env, Q);
                ybar[i] = IloNumArray(env, Q, 0, 1, ILOFLOAT);
                xbar[i] = IloNumArray(env, Q, 0, IloInfinity, ILOINT);
                TD += S[i];
            }

            // ----------------------------------------------------------
            // Objective function: Fixed + variable costs
            // ----------------------------------------------------------
            IloExpr Obj1(env);
            for (IloInt i = 0; i < P; i++)
                for (IloInt j = 0; j < Q; j++)
                    Obj1 += fp[i][j] * y[i][j] + cp[i][j] * x[i][j];
            model.add(IloMinimize(env, Obj1));
            Obj1.end();

            // ----------------------------------------------------------
            // Supply constraints
            // ----------------------------------------------------------
            for (IloInt i = 0; i < P; i++) {
                IloExpr cons1(env);
                for (IloInt j = 0; j < Q; j++) cons1 += x[i][j];
                model.add(cons1 <= S[i]);
                cons1.end();
            }

            // Demand constraints
            for (IloInt j = 0; j < Q; j++) {
                IloExpr cons2(env);
                for (IloInt i = 0; i < P; i++) cons2 += x[i][j];
                model.add(cons2 >= D[j]);
                cons2.end();
            }

            // Linking constraints
            for (IloInt i = 0; i < P; i++)
                for (IloInt j = 0; j < Q; j++)
                    model.add(x[i][j] <= min(S[i], D[j]) * y[i][j]);

            // ----------------------------------------------------------
            // Solve LP relaxation first
            // ----------------------------------------------------------
            IloModel LpRelax(env);
            LpRelax.add(model);
            for (IloInt i = 0; i < P; i++) LpRelax.add(IloConversion(env, y[i], ILOFLOAT));
            IloCplex cplex1(env);
            cplex1.extract(LpRelax);

            int pqcount = 0;
            solveKnapsackAndAddPQcuts(env, model, cplex1, x, y, xbar, ybar, u, v, S, D, TD, P, Q, pqcount);

            // ----------------------------------------------------------
            // Solve the final MIP
            // ----------------------------------------------------------
            IloCplex cplex(model);
            cplex.exportModel("jl1_pQ.lp");
            cplex.setParam(IloCplex::Param::Threads, 1);
          
    /*        cplex.setParam(IloCplex::Covers, -1);
            cplex.setParam(IloCplex::FlowCovers, -1);
            cplex.setParam(IloCplex::MIRCuts, -1);
            cplex.setParam(IloCplex::Param::MIP::Cuts::LiftProj, -1); 
            cplex.setParam(IloCplex::Cliques, -1);
            cplex.setParam(IloCplex::Param::MIP::Cuts::ZeroHalfCut, -1);
            cplex.setParam(IloCplex::Param::MIP::Cuts::Implied, -1);
            cplex.setParam(IloCplex::Param::MIP::Cuts::Gomory, -1);
            cplex.setParam(IloCplex::Param::MIP::Cuts::BQP, -1);
            cplex.setParam(IloCplex::Param::MIP::Cuts::Disjunctive, -1);
            cplex.setParam(IloCplex::Param::MIP::Cuts::PathCut, -1);
            cplex.setParam(IloCplex::Param::MIP::Cuts::GUBCovers, -1);
            cplex.setParam(IloCplex::Param::MIP::Cuts::LocalImplied, -1);
            cplex.setParam(IloCplex::Param::MIP::Cuts::MCFCut, -1);
            cplex.setParam(IloCplex::Param::MIP::Cuts::RLT, -1); */
	            
            cplex.solve();

            cout << "Final objective: " << cplex.getObjValue() << endl;
            cout << "PQ cuts added: " << pqcount << endl;

        } catch (IloException &e) {
            cerr << "Error: " << e << endl;
        } catch (...) {
            cerr << "Unexpected error." << endl;
        }
        env.end();
    }
    return 0;
}

void solveKnapsackAndAddPQcuts(
    IloEnv &env,
    IloModel &model,
    IloCplex &cplex1,
    array2d &x,
    NumBoolVar2DMatrix &y,
    IloNumArray2 &xbar,
    IloNumArray2 &ybar,
    IloBoolVarArray &u,
    IloBoolVarArray &v,
    IloNumArray &S,
    IloNumArray &D,
    const IloNum TD,
    const IloInt P,
    const IloInt Q,
    int &pqcount
) {
    int temp = 1;

    while (true) {
        cplex1.solve();
        cout << "LP objective: " << cplex1.getObjValue() << endl;

        // Store LP solution values for y and x
        for (IloInt i = 0; i < P; i++) {
            for (IloInt j = 0; j < Q; j++) {
                ybar[i][j] = cplex1.getValue(y[i][j]);
                xbar[i][j] = cplex1.getValue(x[i][j]);
            }
        }

        // ----------------------------------------------------------
        // Solve knapsack separation problem
        // ----------------------------------------------------------
        IloModel knapsack(env);
        IloExpr Obj2(env);
        for (IloInt i = 0; i < P; i++) {
            for (IloInt j = 0; j < Q; j++) {
                Obj2 += ybar[i][j] * u[i] * v[j];
            }
        }
        knapsack.add(IloMinimize(env, Obj2));
        Obj2.end();

        // Knapsack capacity constraints
        IloExpr lhs(env);
        for (IloInt i = 0; i < P; i++) lhs += S[i] * u[i];
        IloExpr rhs(env);
        for (IloInt j = 0; j < Q; j++) rhs += D[j] * v[j];
        knapsack.add(lhs >= TD - rhs + 1);
        lhs.end();
        rhs.end();

        IloCplex cplex2(knapsack);
        temp++;
        if (temp == 2) cplex2.exportModel("Q2.lp");
        cplex2.solve();
        cout << "Knapsack objective: " << cplex2.getObjValue() << endl;

        // If cut is found, add PQ cut; else, exit loop
        if (cplex2.getObjValue() < 1) {
            IloNumArray uval(env, P);
            IloNumArray vval(env, Q);
            for (IloInt i = 0; i < P; i++) uval[i] = cplex2.getValue(u[i]) < 0.001 ? 0 : 1;
            for (IloInt j = 0; j < Q; j++) vval[j] = cplex2.getValue(v[j]) < 0.001 ? 0 : 1;

            AddPQCut(env, model, y, uval, vval, P, Q);
            pqcount++;
        } else {
            break;
        }
    }
}

void AddPQCut(
    IloEnv &env,
    IloModel &model,
    const NumBoolVar2DMatrix &y,
    const IloNumArray &uval,
    const IloNumArray &vval,
    const IloInt P,
    const IloInt Q
) {
    IloExpr pqcut(env);

    for (IloInt i = 0; i < P; i++)
        for (IloInt j = 0; j < Q; j++)
            pqcut += uval[i] * vval[j] * y[i][j];

    model.add(pqcut >= 1);
    pqcut.end();
}
