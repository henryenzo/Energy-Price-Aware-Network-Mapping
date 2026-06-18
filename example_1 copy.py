#!/usr/bin/env python3

# Copyright 2026, Gurobi Optimization, LLC

# This example formulates and solves the following simple MIP model:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

import gurobipy as gp
from gurobipy import GRB
from model_class import NetworkMapping

try:
    # Create a new model
    m = gp.Model("mip1")

    # Create variables
    phi_22 = m.addVar(vtype=GRB.BINARY, name="phi_22")
    phi_23 = m.addVar(vtype=GRB.BINARY, name="phi_23")

    # Set objective
    m.setObjective(phi_22 + phi_23, GRB.MAXIMIZE)

    # Add constraint: phi_22 + 2 phi_23 <= 4
    m.addConstr(phi_22 + 2 * phi_23 <= 4, "c0")

    # Add constraint: phi_22 + phi_23 >= 1
    m.addConstr(phi_22 + phi_23 >= 1, "c1")

    # Optimize model
    m.optimize()

    for v in m.getVars():
        print(f"{v.VarName} {v.X:g}")

    print(f"Obj: {m.ObjVal:g}")

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")