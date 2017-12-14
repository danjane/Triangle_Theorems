import numpy as np
import pandas as pd
import cvxpy as cvx
import pylab

df = pd.DataFrame.from_csv('test.csv', index_col=None, sep=',')
(nObs, nVar) = df.shape

M = np.ones([nObs, nVar+1])
M[:, :-1] = df.values

solutions = np.ones([nVar+1, nVar])

for i in range(nVar):
    idx = list(range(nVar+1))
    del idx[i]
    c = M[:, i]
    A = M[:, idx]

    x = cvx.Variable(nVar)
    constraints = [A * x + c <= 1e-4]
    obj = cvx.Minimize(cvx.norm(x, 1))
    prob = cvx.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    if prob.status == cvx.OPTIMAL:
        solutions[idx, i] = x.value.flatten()
    else:
        solutions[:, i] = np.NaN

chk = np.dot(M, solutions)
idx = np.amax(abs(chk), axis=0) < 1e-6
solutions = solutions[:, idx]

print solutions

chk = np.dot(M, solutions)
pylab.plot(chk)
pylab.show()