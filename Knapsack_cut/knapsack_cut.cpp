#include<stdio.h>
#include<iostream>

#include <ilcplex/ilocplex.h>
#include <ilconcert/ilosys.h>

using namespace std;

ILOSTLBEGIN;
typedef IloArray<IloNumArray> Num2DMatrix;
typedef IloArray<Num2DMatrix> Num3DMatrix;//3D array of Num
typedef IloArray<IloBoolVarArray> NumBoolVar2DMatrix; // 2D array of NumVar
typedef IloArray<NumBoolVar2DMatrix> NumBoolVar3DMatrix;//3D array of Var
typedef IloArray<IloNumVarArray> array2d;//Creating a 2d array of x variables
typedef IloArray<array2d> array3d;//Creating a 3d array of x variables

int main(int argc, char **argv) 
{
	clock_t t1,t2;
	t1=clock();
	IloEnv env;
	
	
	IloNum C; // Knapsack Capacity
	IloNumArray W(env);	// weight of items
	IloNumArray P(env);	// profit of each item
	IloNum N;
	
	ifstream fin;
	const char* filename = "P_Slide.csv";
	fin.open(filename);
	fin >> C >> W >> P;
	cout << "Reading Data from the file - "<<filename<<endl;
	fin.close();
	N = W.getSize();
	cout <<"N: "<<N<<endl;


	try
	{
		IloModel model(env);
		
		IloBoolVarArray x(env,N);
		IloBoolVarArray y(env, N);// variable for separation problem
		
		IloNumArray xbar(env,N);
		IloNumArray ybar(env, N);
		
		//-----------------------------------Objective function---------------------------------------
		IloExpr obj(env);
		for(IloInt j=0; j<N; j++)
		{
			obj += P[j]*x[j];

		}		
		
		model.add(IloMaximize(env,obj));
		obj.end();
		
		//--------------------------------------------------constraint 1---------------------------------------
		IloExpr cons1(env);
		for (IloInt j = 0; j < N; j++)
		{
			cons1 += W[j] * x[j];
		}
		model.add(cons1 <= C);
		cons1.end();

		// Code for LP relaxation
		//---------------------------------
		IloModel LpRelax(env);
		LpRelax.add(model);
		for (IloInt j = 0; j < N; j++)
		{
			LpRelax.add(IloConversion(env, x[j], ILOFLOAT));
		}

		IloCplex cplex1(env);
		cplex1.extract(LpRelax);
		int count = 0; // no of cover cuts added.
		cplex1.solve();
		cout << "LP relaxation objective value: " << cplex1.getObjValue() << endl;
		//-------------------------------------
		
		// Branch and cut

		while (1)
		{
			cplex1.solve();
			cout << "total cost: " << cplex1.getObjValue() << endl;
			cout << endl;


			for (IloInt j = 0; j < N; j++)
			{
					xbar[j] = cplex1.getValue(x[j]);
					cout << xbar[j] << endl;

			}

			//-----------------------------------Objective function---------------------------------------
			IloModel sepmodel(env);

			IloExpr ObjSep(env);
			for (IloInt j = 0; j < N; j++)
			{
				ObjSep += (1 - xbar[j]) * y[j];
			}

			sepmodel.add(IloMinimize(env, ObjSep));
			ObjSep.end();

			//--------------------------------------------------constraint 1---------------------------------------
			IloExpr sepcons1(env);
			for (IloInt j = 0; j < N; j++)
			{
				sepcons1 += W[j] * y[j];
			}
			sepmodel.add(sepcons1 >= C+1);
			sepcons1.end();

			IloCplex cplex2(sepmodel);
			cplex2.exportModel("sep1.lp");
			cplex2.solve();
			float Sep_obj = cplex2.getObjValue();

			cout << "separation objective function " << cplex2.getObjValue() << endl;
			if (Sep_obj < 1)
			{
				count += 1;
				for (IloInt j = 0; j < N; j++)
				{
					if (cplex2.getValue(y[j]) < 0.01)
						ybar[j] = 0;

					else
						ybar[j] = 1;
				}

				int card = 0; // cardinality
				for (IloInt j = 0; j < N; j++)
				{
					card += ybar[j];
				}
				cout << "cardinality of cover" << endl;
				// Adding cover cut to the kanpasck problem (model)

				IloExpr covercut(env);
				for (IloInt j = 0; j < N; j++)
				{
					covercut += ybar[j] * x[j];

				}
				model.add(covercut <= card - 1);
				covercut.end();
			}
			else
			{
				
				break;
			}
				
				//cout << endl;
		}
		cout << " No of cover cuts added " << count << endl;
		
		
		IloCplex cplex(model);
		
		cplex.setParam(IloCplex::Param::Preprocessing::Presolve, 0);
		cplex.setParam(IloCplex::Param::MIP::Strategy::Search, 0);
		cplex.setParam(IloCplex::Covers, -1);
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
		cplex.setParam(IloCplex::Param::MIP::Cuts::RLT, -1);


		cplex.solve();
		cout<<"total cost: "<< cplex.getObjValue() << endl;
		cplex.exportModel("Knapsack_cut.lp");
		
		cout << "No. of B&B Tree Nodes = " << cplex.getNnodes() << endl;

	}
	catch(IloException& e)
	{
		cerr<<"error: "<<e<< endl;
	}
	catch(...)
	{
		cerr<<"unexpected error occur"<<endl;
	}
	env.end();
	return 0;
}
