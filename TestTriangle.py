import Geometry

# Do a few general constructions
triangle = Geometry.GeometricCollection()
triangle.make_triangle()
print triangle.tasks
triangle.do_all_tasks()
print triangle.tasks

for i in range(4):
    triangle.do_all_tasks()
    triangle.plot_constructions()
