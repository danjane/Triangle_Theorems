import numpy as np
import pandas as pd
import cvxpy as cvx
# import pylab
import Geometry

def solutions_from_data(df):
    (nObs, nVar) = df.shape
    geometric_objects = df.columns

    M = np.ones([nObs, nVar + 1])
    M[:, :-1] = df.values

    solutions = []

    regressors = np.full((nVar, 1), True, dtype=bool)
    while any(regressors):
        i = np.where(regressors)[0][0]
        regressors[i] = False
        print "Finding relations involving: {:s}".format(geometric_objects[i])

        idx = list(range(nVar + 1))
        del idx[i]
        sol = np.ones(nVar + 1)

        c = M[:, i]
        A = M[:, idx]

        x = cvx.Variable(nVar)
        constraints = [A * x + c <= 1e-7, A * x + c >= -1e-7]
        obj = cvx.Minimize(cvx.norm(x[:-1], 1))
        prob = cvx.Problem(obj, constraints)
        prob.solve()  # Returns the optimal value.
        if prob.status == cvx.OPTIMAL:
            sol[idx] = x.value.flatten()
            sol[abs(sol) < 1e-6] = 0.
            sol[abs(sol - 1) < 1e-6] = 1.

            regressors[abs(sol[:-1]) > 1e-6] = False

            solutions.append(sol)

    solutions = np.stack(solutions, axis=1)

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