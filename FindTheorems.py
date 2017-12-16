import numpy as np
import pandas as pd
import cvxpy as cvx
import pylab
import scipy.linalg.interpolative as sli

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
    constraints = [A * x + c <= 1e-7, A * x + c >= -1e-7]
    obj = cvx.Minimize(cvx.norm(x[:-1], 1))
    prob = cvx.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    if prob.status == cvx.OPTIMAL:
        solutions[idx, i] = x.value.flatten()
    else:
        solutions[:, i] = np.NaN

chk = np.dot(M, solutions)

# print solutions
# pylab.plot(chk)
# pylab.show()

idx = np.amax(abs(chk), axis=0) < 1e-6
solutions = solutions[:, idx]

pylab.pcolor(solutions, vmin=np.nanmin(solutions), vmax=np.nanmax(solutions))
pylab.colorbar()
pylab.show()

k, idx, proj = sli.interp_decomp(solutions, 1e-6)
solutions = sli.reconstruct_skel_matrix(solutions, k, idx)
solutions[abs(solutions)<1e-6] = 0.
solutions[abs(solutions-1)<1e-6] = 1.

print solutions

