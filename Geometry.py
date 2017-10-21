import numpy as np

class Task:
    """A geometric construction."""

    def __init__(self, obj1, obj2):
        """Initializes the data."""
        self.doable = False

        if (obj1.name == 'Point') & (obj2.name == 'Point'):
            self.obj1 = obj1
            self.obj2 = obj2
            self.type = 'p2p'
            self.text = "Join two points with a line ({} and {})".format(id(obj1), id(obj2))
            self.doable = True


class GeometricCollection:
    """Represents a collection of geometric objects."""

    def __init__(self):
        """Initializes the data."""
        self.population = []
        self.tasks = []

        print("(Initializing GC = {})".format(id(self)))

    def point(self, x, y):
        p = Point(x, y)
        self.add_obj(p)

    def line(self, x, y, theta):
        p = Line(x, y, theta)
        self.add_obj(p)

    def add_obj(self, new_obj):
        for obj in self.population:
            task = Task(obj, new_obj)
            if task.doable:
                self.tasks.append(task)

        self.population.append(new_obj)

    def connect_points(self):
        new_tasks = []
        now_tasks = []
        for task in self.tasks:
            if task.type == 'p2p':
                now_tasks.append(task)
            else:
                new_tasks.append(task)
        self.tasks = new_tasks

        for task in now_tasks:
            p = task.obj1
            q = task.obj2
            self.line(p.x, p.y, np.arctan2(p.y-q.y, p.x-q.x))

    def what_tasks(self):
        """Prints the current population."""
        print("We have the following {:d} tasks outstanding:\n".format(len(self.tasks)))
        for task in self.tasks:
            print("{}\n".format(task.text))


class Geometric:
    """Represents an abstract geometric notion, with a name."""

    def __init__(self, name):
        """Initializes the data."""
        self.name = name
        print("(Initializing {})".format(self.name))


class Point(Geometric):
    """Represents a point (x,y)."""

    def __init__(self, x, y):
        Geometric.__init__(self, 'Point')
        self.x = x
        self.y = y


class Line(Geometric):
    """Represents a line through (x,y) at angle theta."""

    def __init__(self, x, y, theta):
        Geometric.__init__(self, 'Line')
        self.x = x
        self.y = y
        self.theta = theta

triangle = GeometricCollection()
triangle.point(0.0, 0.0)
triangle.point(1.0, 0.0)
triangle.point(1.0, 1.0)
triangle.connect_points()
triangle.what_tasks()