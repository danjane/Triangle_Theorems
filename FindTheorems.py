import numpy as np
import pandas as pd
import cvxpy as cvx
# import pylab
import scipy.linalg.interpolative as sli
import Geometry


def solutions_from_data(df):
    (nObs, nVar) = df.shape
    geometric_objects = df.columns

    M = np.ones([nObs, nVar + 1])
    M[:, :-1] = df.values

    solutions = []

    # TODO: don't need to check a regressor once it has a non-zero value in some optimal solution
    for i in range(nVar):
        print "Finding relations involving: {:s}".format(geometric_objects[i])

        idx = list(range(nVar + 1))
        del idx[i]
        solution = np.ones(nVar + 1)

        c = M[:, i]
        A = M[:, idx]

        x = cvx.Variable(nVar)
        constraints = [A * x + c <= 1e-7, A * x + c >= -1e-7]
        obj = cvx.Minimize(cvx.norm(x[:-1], 1))
        prob = cvx.Problem(obj, constraints)
        prob.solve()  # Returns the optimal value.
        if prob.status == cvx.OPTIMAL:
            solution[idx] = x.value.flatten()
            solutions.append(solution)

    solutions = np.stack(solutions, axis=1)

    # pylab.pcolor(solutions, vmin=np.nanmin(solutions), vmax=np.nanmax(solutions))
    # pylab.colorbar()
    # pylab.show()

    k, idx, proj = sli.interp_decomp(solutions, 1e-6)
    solutions = sli.reconstruct_skel_matrix(solutions, k, idx)
    solutions[abs(solutions) < 1e-6] = 0.
    solutions[abs(solutions - 1) < 1e-6] = 1.

    return solutions


def determine_type_of_relation(solution, constructs):
    c = -solution[-1]
    solution = solution[:-1]
    constructs = constructs[solution != 0.]

    if all([construct[0] == Geometry.angle_from_two_lines for construct in constructs]):
        print "angles add up to {}".format(c)
    elif (len(constructs) == 1) and (constructs[0][0] == Geometry.connect_points):
        if c == 0:
            print "points concurrent"
        else:
            print "points a fixed distance"
    else:
        print "unknown relation!!"


print "/nLoading: 'test.pickle'\n"
# df = pd.DataFrame.from_csv('test.csv', index_col=None, sep=',')
df = pd.read_pickle('test.pickle')

solutions = solutions_from_data(df)
print solutions

for i in range(solutions.shape[1]):
    determine_type_of_relation(solutions[:, i], df.columns)