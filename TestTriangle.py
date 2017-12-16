import Geometry

# Do a few general constructions
triangle = Geometry.GeometricCollection()
triangle.make_triangle()
for i in range(5):
    triangle.do_all_tasks()
    triangle.plot_constructions()
