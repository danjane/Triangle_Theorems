import Geometry
import numpy as np
import pandas as pd
import time

# Do a few general constructions
triangle = Geometry.GeometricCollection()
triangle.make_triangle()
for i in range(5):
    triangle.do_all_tasks()
    # triangle.plot_constructions()

# triangle.show_data()
# print "There are {:d} geometric objects".format(len(triangle.objects))
# print "There were {:d} tasks performed:".format(len(triangle.tasks_done))
# print triangle.tasks_done

# Find a potentially interesting construction
min_dist = np.Inf
for k, v in triangle.data.iteritems():
    if v < min_dist:
        min_dist = v
        min_key = k

print "\nClosest points at a distance of {:g}".format(min_dist)
print "Occurs for construction {}".format(min_key)

# Now recreate this construction for general triangles to see if the rule holds
start = time.clock()
dfs = []
for i in range(1000):
    random_triangle = Geometry.RandomTriangle()
    random_triangle.construct_point(min_key)

    # random_triangle.plot_constructions()
    # random_triangle.show_data()
    dfs.append(pd.DataFrame.from_dict(random_triangle.data, orient='index').T)
df = pd.concat(dfs)
end = time.clock()

print "\n1000 constructions complete, avg time {} secs.".format((end - start)/1000.)


# df[df==0] = np.nan # necessary?
# df = df.dropna(axis=1)

df.to_csv('test.csv', na_rep='NaN', index=False)
print "Data written to test.csv"
