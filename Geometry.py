import numpy as np
import pylab


class GeometricCollection:
    """Represents a collection of geometric objects."""

    def __init__(self):
        """Initializes the data."""
        self.population = 0
        self.objects = []
        self.parents = {}
        self.data = {}
        self.tasks = set()  # potential parents

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
        self.tasks.add((x.number,))
        self.data[x.number] = np.rad2deg(x.theta)
        # Also add complement angle
        # x = Angle(x.theta + x.alpha - np.pi, x.alpha, x.point) #hmmm... clearly needs a think
        # self.add_obj(x)
        # self.data.append(x.theta)
        return x

    def add_obj(self, new_obj):
        new_num = self.population
        new_obj.number = new_num

        # This new object can have new kids!
        for old_num in range(new_num):
            if ((old_num - new_num) % 3) == 2:
                self.tasks.add((new_num, old_num))
            else:
                self.tasks.add((old_num, new_num))

        self.objects.append(new_obj)
        self.population += 1
        return new_obj

    def connect_points(self, p, q):
        dy = p.y-q.y
        dx = p.x-q.x
        l = self.line(p.x, p.y, np.arctan2(dy, dx), [p, q])
        self.data[l.number] = np.sqrt(dx**2 + dy**2)

    def intersect_lines(self, l1, l2):
        p = set(l1.parents).intersection(l2.parents)

        if p:
            p = p.pop()
            p = self.objects[p]

        else:

            # Check for parallel lines?
            if l1.theta == l2.theta:
                if l1.theta == np.arctan2(l1.y - l2.y, l1.x - l2.x):
                    print("Lines are equal!! ({} and {})".format(id(l1), id(l2)))
                else:
                    print("Lines are parallel!! ({} and {})".format(id(l1), id(l2)))

            c1 = np.tan(l1.theta)
            c2 = np.tan(l2.theta)

            x = (l2.y - l1.y - c2 * l2.x + c1 * l1.x) / (c1 - c2)
            y = (c1 * l2.y - c2 * l1.y - c1 * c2 * l2.x + c1 * c2 * l1.x) / (c1 - c2)

            p = self.point(x, y, [l1, l2])

        # Now create an angle from this intersection
        self.angle(l1, l2, p)

    def trisect_angle(self, angle):
        p = angle.point
        self.line(p.x, p.y, angle.alpha + angle.theta/3., [p])
        self.line(p.x, p.y, angle.alpha + angle.theta*2./3., [p])

    def do_all_tasks(self):
        current_tasks = set(self.tasks)
        self.tasks = set()
        for task in current_tasks:
            self.do_task(task)

    def do_task(self, parents):
        if len(parents) == 1:
            self.do_task1(parents)
        elif len(parents) == 2:
            self.do_task2(parents)
        else:
            print("something gone wrong!!\n")

    def do_task2(self, parents):
        obj1 = self.objects[parents[0]]
        obj2 = self.objects[parents[1]]
        if (obj1.name == 'Point') and (obj2.name == 'Point'):
            self.connect_points(obj1, obj2)
        elif (obj1.name == 'Line') and (obj2.name == 'Line'):
            self.intersect_lines(obj1, obj2)

    def do_task1(self, parents):
        obj1 = self.objects[parents[0]]
        if obj1.name == 'Angle':
            self.trisect_angle(obj1)

    def what_tasks(self):
        """Prints the current population."""
        print("We have the following {:d} tasks outstanding:\n".format(len(self.tasks)))
        for task in self.tasks:
            print("{}\n".format(task.text))

    def show_data(self):
        """Prints the current population."""
        print("{}\n".format(self.data))

    def plot_constructions(self):

        for obj in reversed(self.objects):
            obj.plot()

        # pylab.title('Probability distribution of anharmonic oscillator with beta=' + str(beta))
        # pylab.xlabel('position')
        # pylab.ylabel('probability')
        # pylab.legend(['matrix squaring', 'path sampled'])
        pylab.xlim(-1., 2.)
        pylab.ylim(-1., 2.)
        pylab.show()
        pylab.axis('scaled')


class Geometric:
    """Represents an abstract geometric notion, with a name."""

    def __init__(self, name):
        """Initializes the data."""
        self.name = name
        self.parents = []
        self.number = -1
        print("(Initializing {})".format(self.name))

    def plot(self):
            pass


class Point(Geometric):
    """Represents a point (x,y)."""

    def __init__(self, x, y, lines=None):
        Geometric.__init__(self, 'Point')
        self.x = x
        self.y = y
        if lines is None:
            lines = []
        self.parents = lines
        print("Point x:{}, y:{}:".format(x, y))

    def plot(self):
        pylab.plot(self.x, self.y, 'ro')


class Line(Geometric):
    """Represents a line through (x,y) at angle theta."""

    def __init__(self, x, y, theta, points=None):
        Geometric.__init__(self, 'Line')
        self.x = x
        self.y = y
        self.theta = theta
        if points is None:
            points = []
        self.parents = [p.number for p in points]

    def plot(self):
        pylab.plot(
            [self.x + edge * np.cos(self.theta) for edge in [-2, 2]],
            [self.y + edge * np.sin(self.theta) for edge in [-2, 2]],
            linewidth=1, color='g')


class Angle(Geometric):
    """Represents an angle theta (inclined at angle alpha) at a point."""

    def __init__(self, line1, line2, point):
        Geometric.__init__(self, 'Angle')

        theta = line1.theta-line2.theta - np.pi
        alpha = line2.theta

        self.theta = theta % (2*np.pi)
        self.alpha = alpha % (2*np.pi)
        self.point = point
        self.parents = [line1.number, line2.number]

    def plot(self):
        p = self.point
        xs = [self.alpha + self.theta*x/99. for x in range(100)]
        pylab.plot(
            [p.x + 0.1 * np.cos(x) for x in xs],
            [p.y + 0.1 * np.sin(x) for x in xs],
            linewidth=.5, color='c')

triangle = GeometricCollection()
triangle.point(0., 0.)
triangle.point(1., 0.1)
triangle.point(1.1, 1.)
triangle.do_all_tasks()
triangle.do_all_tasks()
triangle.do_all_tasks()
triangle.do_all_tasks()

triangle.show_data()
print len(triangle.tasks)
print min([abs(i-60) for i in triangle.data])
triangle.plot_constructions()
