import numpy as np
import pylab

MAX_OBJECTS = 999


def connect_points((p, q)):
    dy = p.y - q.y
    dx = p.x - q.x
    l = Line((connect_points, (p.name, q.name)), p.x, p.y, np.arctan2(dy, dx), set([p, q]))
    p.lines.append(l)
    q.lines.append(l)
    return l, np.sqrt(dx**2 + dy**2)


def intersect_lines((l1, l2)):
    p = l1.points.intersection(l2.points)

    if p:
        return 0, np.nan

    else:

        # Check for parallel lines?
        if ((l1.theta - l2.theta) % (2 * np.pi)) == 0:
            if l1.theta == np.arctan2(l1.y - l2.y, l1.x - l2.x):
                print("Lines are equal!! ({} and {})".format(id(l1), id(l2)))
            else:
                print("Lines are parallel!! ({} and {})".format(id(l1), id(l2)))

        c1 = np.cos(l1.theta)
        c2 = np.cos(l2.theta)
        s1 = np.sin(l1.theta)
        s2 = np.sin(l2.theta)

        x = (c1 * c2 * (l2.y - l1.y) - c1 * s2 * l2.x + s1 * c2 * l1.x) / (s1 * c2 - s2 * c1)
        y = (s1 * c2 * l2.y - s2 * c1 * l1.y - s1 * s2 * l2.x + s1 * s2 * l1.x) / (s1 * c2 - s2 * c1)

        p = Point((intersect_lines, (l1.name, l2.name)), x, y, [l1, l2])

        l1.points.add(p)
        l2.points.add(p)

        return p, np.nan


def angle_from_two_lines((l1, l2)):
    p = l1.points.intersection(l2.points)
    p = p.pop()
    a = Angle((angle_from_two_lines, (l1.name, l2.name)), l1, l2, p)
    return a, a.theta


def bisect_angle((angle)):
    if type(angle) is tuple:
        angle = angle[0]
    l = Line((bisect_angle, (angle.name,)),
             angle.x, angle.y, angle.alpha + angle.theta/2., set([angle.point]))
    return l, np.nan


class GeometricCollection(object):
    """Represents a collection of geometric objects."""

    def __init__(self):
        """Initializes the data."""
        self.population = 0
        self.objects = []
        self.data = {}
        self.tasks = []
        self.tasks_done = []

    def point(self, *arg, **kw):
        x = Point(*arg, **kw)
        self.add_obj(x)
        return x

    def line(self, *arg, **kw):
        x = Line(*arg, **kw)
        self.add_obj(x)
        return x

    def angle(self, *arg, **kw):
        x = Angle(*arg, **kw)
        x = self.add_obj(x)
        return x

    def add_obj(self, new_obj):
        new_num = self.population
        new_obj.number = new_num

        # This new object can have new kids!
        if len(self.objects) < MAX_OBJECTS:
            self.tasks.append((new_num,))
            for old_num in range(new_num):
                if ((old_num - new_num) % 3) == 2:
                    self.tasks.append((new_num, old_num))
                else:
                    self.tasks.append((old_num, new_num))

        self.objects.append(new_obj)
        self.population += 1
        return new_obj

    def two_points(self, p, q):
        l, distance = connect_points((p, q))
        self.add_obj(l)
        self.data[l.name] = distance

    def two_lines(self, l1, l2):
        p, distance = intersect_lines((l1, l2))
        if p:
            self.add_obj(p)
        a, angle = angle_from_two_lines((l1, l2))
        self.add_obj(a)
        self.data[a.name] = angle

    def one_angle(self, angle):
        a, distance = bisect_angle(angle)
        self.add_obj(a)

    def trisect_angle(self, angle):
        p = angle.point
        self.line(p.x, p.y, angle.alpha + angle.theta/3., [p])
        self.line(p.x, p.y, angle.alpha + angle.theta*2./3., [p])

    def do_all_tasks(self):
        current_tasks = self.tasks
        self.tasks = []
        for task in current_tasks:
            if len(self.objects) > MAX_OBJECTS:
                break
            task = self.do_task(task)
            if task is not None:
                self.tasks_done.append(task)

        print "There are {:d} tasks outstanding".format(len(self.tasks))

    def do_task(self, parents):
        if len(parents) == 1:
            task = self.do_task1(parents)
        elif len(parents) == 2:
            task = self.do_task2(parents)
        else:
            print("something gone wrong!!\n")
        return task

    def do_task2(self, parents):
        obj1 = self.objects[parents[0]]
        obj2 = self.objects[parents[1]]
        task = None
        if isinstance(obj1, Point) and isinstance(obj2, Point):
            self.two_points(obj1, obj2)
            task = (obj1.name, obj2.name)
        elif isinstance(obj1, Line) and isinstance(obj2, Line):
            self.two_lines(obj1, obj2)
            task = (obj1.name, obj2.name)
        return task

    def do_task1(self, parents):
        obj1 = self.objects[parents[0]]
        task = None
        if isinstance(obj1, Angle):
            # self.trisect_angle(obj1)
            self.one_angle(obj1)
            task = (obj1.name,)
        return task

    def what_tasks(self):
        """Prints the current population."""
        print("We have the following {:d} tasks outstanding:\n".format(len(self.tasks)))
        for task in self.tasks:
            print("{}\n".format(task.text))

    def make_triangle(self):
        self.point('A', 0., 0.)
        self.point('B', 1., 0.)
        self.point('C', 1., 1.)

    def show_data(self):
        """Prints the current population."""
        print("{}\n".format(self.data))

    def plot_constructions(self):

        for obj in reversed(self.objects):
            obj.plot()

        pylab.xlim(-1., 2.)
        pylab.ylim(-1., 2.)
        pylab.show()
        pylab.axis('scaled')


class RandomTriangle(GeometricCollection):

    def __init__(self):
        """Initializes the data."""
        self.objects = {}
        self.data = {}
        self.make_random_triangle()

    def make_random_triangle(self):
        self.point('A', -0.5, 0.)
        self.point('B', 0.5, 0.)
        self.point('C', np.random.normal()/4, np.abs(np.random.normal()))

    def add_obj(self, *args):
        new_obj = args[0]
        self.objects[new_obj.name] = new_obj
        if len(args) > 1:
            data = args[1]
            if not(np.isnan(args[1])):
                self.data[new_obj.name] = data

    def construct_point(self, name):
        if name in self.objects:
            return self.objects[name]
        else:
            func = name[0]
            points = tuple(self.construct_point(point) for point in name[1])
            new_obj, data = func(points)
            self.add_obj(new_obj, data)
            return new_obj

    def plot_constructions(self):

        for name, obj in self.objects.iteritems():
            obj.plot()

        pylab.xlim(-1., 1.)
        pylab.ylim(-1., self.objects['C'].y*2)
        pylab.show()
        pylab.axis('scaled')


class Geometric(object):
    """Represents an abstract geometric notion, with a name."""

    def __init__(self, name):
        """Initializes the data."""
        self.name = name
        self.number = -1

    def plot(self):
            pass


class Point(Geometric):
    """Represents a point (x,y)."""

    def __init__(self, name, x, y, lines=None):
        Geometric.__init__(self, name)
        self.x = x
        self.y = y
        if lines is None:
            lines = []
        self.lines = [l.number for l in lines]

    def plot(self):
        pylab.plot(self.x, self.y, 'ro')


class Line(Geometric):
    """Represents a line through (x,y) at angle theta."""

    def __init__(self, name, x, y, theta, points=set()):
        Geometric.__init__(self, name)
        self.x = x
        self.y = y
        self.theta = theta % (2*np.pi)
        self.points = points

    def plot(self):
        pylab.plot(
            [self.x + edge * np.cos(self.theta) for edge in [-2, 2]],
            [self.y + edge * np.sin(self.theta) for edge in [-2, 2]],
            linewidth=1, color='g')


class Angle(Geometric):
    """Represents an angle theta (inclined at angle alpha) at a point."""

    def __init__(self, name, line1, line2, point):
        Geometric.__init__(self, name)

        theta = line1.theta-line2.theta - np.pi
        alpha = line2.theta

        self.theta = theta % (2*np.pi)
        self.alpha = alpha % (2*np.pi)
        self.x = point.x
        self.y = point.y
        self.point = point

    def plot(self):
        xs = [self.alpha + self.theta*x/99. for x in range(100)]
        pylab.plot(
            [self.x + 0.1 * np.cos(x) for x in xs],
            [self.y + 0.1 * np.sin(x) for x in xs],
            linewidth=.5, color='c')


