
import gurobipy as gp
from gurobipy import GRB

def solve_ch10(data):
    Q2 = data["Q2"]
    mu2 = data["mu2"]
    lb = data["lb"]
    ub = data["ub"]
    w_b = data["w_b"]
    cost = data["cost"]
    n = data["n"]
    target_return = data["target_return"]
    vartypes = data["vartypes"]
    tickers = data["tickers"]

    model = gp.Model("CH10_TransactionCostPortfolio")
    model.setParam("OutputFlag", 0)

    x = model.addVars(6 * n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    for i in range(4 * n, 6 * n):
        x[i].vtype = GRB.BINARY

    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 1)

    for i in range(n):
        model.addConstr(x[i] - x[n + i] + x[2 * n + i] - w_b[i] == 0)
        model.addConstr(x[3 * n + i] == w_b[i])
        model.addConstr(x[n + i] <= 1 * x[4 * n + i])
        model.addConstr(x[n + i] >= 0 * x[4 * n + i])
        model.addConstr(x[2 * n + i] <= 1 * x[5 * n + i])
        model.addConstr(x[2 * n + i] >= 0 * x[5 * n + i])

    model.addConstr(
        gp.quicksum(data["mu"][i] * x[i] for i in range(n)) -
        gp.quicksum(cost[i] * (x[n + i] + x[2 * n + i]) for i in range(n))
        >= target_return
    )

    lambda_risk = 2
    quad_expr = gp.QuadExpr()
    for i in range(6 * n):
        for j in range(6 * n):
            if Q2[i, j] != 0:
                quad_expr += Q2[i, j] * x[i] * x[j]

    linear_expr = gp.quicksum(mu2[i] * x[i] for i in range(6 * n))
    model.setObjective(linear_expr - lambda_risk * quad_expr, GRB.MAXIMIZE)
    model.optimize()

    return {tickers[i]: x[i].X for i in range(n) if x[i].X > 0.001}, model.ObjVal
