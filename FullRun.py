import Geometry
import numpy as np
import pandas as pd
import time
import FindTheorems

# Do a few general constructions
triangle = Geometry.GeometricCollection()
triangle.make_triangle()
while triangle.do_all_tasks():
    triangle.forget_some_tasks(100)
print "Finished randomly constructing"

# Now recreate this construction for general triangles to see if the rule holds
start = time.clock()
dfs = []
fails = 0
for i in range(10000):
    random_triangle = Geometry.RandomTriangle()
    try:
        random_triangle.construct_point(triangle.objects[-300].name)
        dfs.append(pd.DataFrame.from_dict(random_triangle.data, orient='index').T)
    except AttributeError:
        fails += 1
df = pd.concat(dfs)
end = time.clock()

print "\n{} constructions complete".format(df.shape[0])
print "Total time {} secs.".format(end - start)

solutions = FindTheorems.solutions_from_data(df)
print np.stack(solutions, axis=1)
print '\n'

for s in solutions:
    FindTheorems.determine_type_of_relation(s, df.columns)
