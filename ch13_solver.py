
import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_ch13(mu, sigma, risk_aversion, leverage_long=1, leverage_short=1):
    n = len(mu)
    N = 5 * n

    model = gp.Model("CH13_MarketNeutralPortfolio")
    model.setParam("OutputFlag", 0)

    x = model.addVars(N, lb=-2.0, ub=2.0, vtype=GRB.CONTINUOUS, name="x")
    for i in range(3 * n, N):
        x[i].vtype = GRB.BINARY

    # Q matrix
    Q = np.zeros((N, N))
    Q[:n, :n] = sigma
    quad_expr = gp.QuadExpr()
    for i in range(N):
        for j in range(N):
            if Q[i, j] != 0:
                quad_expr += Q[i, j] * x[i] * x[j]

    # Linear term
    mu2 = np.concatenate([mu, np.zeros(4 * n)])
    linear_expr = gp.quicksum(mu2[i] * x[i] for i in range(N))

    # Objective
    model.setObjective(linear_expr - risk_aversion * quad_expr, GRB.MAXIMIZE)

    # Equality Constraints
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 0)  # Dollar neutrality

    for i in range(n):  # w = w+ - w-
        model.addConstr(x[i] == x[n + i] - x[2 * n + i])

    model.addConstr(gp.quicksum(x[n + i] for i in range(n)) == leverage_long)
    model.addConstr(gp.quicksum(x[2 * n + i] for i in range(n)) == leverage_short)

    # Binary Activation Constraints
    for i in range(n):
        model.addConstr(x[n + i] <= 1 * x[3 * n + i])
        model.addConstr(x[n + i] >= 0 * x[3 * n + i])
        model.addConstr(x[2 * n + i] <= 1 * x[4 * n + i])
        model.addConstr(x[2 * n + i] >= 0 * x[4 * n + i])
        model.addConstr(x[3 * n + i] + x[4 * n + i] <= 1)  # only one binary active

    model.optimize()

    return {f"Stock{i+1}": x[i].X for i in range(n) if abs(x[i].X) > 0.001}, model.ObjVal
