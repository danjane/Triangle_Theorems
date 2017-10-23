import numpy as np
import pylab

class Task:
    """A geometric construction."""

    def __init__(self, obj1, obj2):
        """Initializes the data."""
        self.obj1 = obj1
        self.obj2 = obj2
        self.doable = False

        if (obj1.name == 'Point') and (obj2.name == 'Point'):
            self.type = 'p2p'
            self.text = "Join two points with a line ({} and {})".format(id(obj1), id(obj2))
            self.doable = True

        if (obj1.name == 'Line') and (obj2.name == 'Line'):
            self.type = 'lx'
            self.text = "Find point of intersection ({} and {})".format(id(obj1), id(obj2))
            self.doable = True

        if (obj1.name == 'Point') and (obj2.name == 'Angle') and (obj2.point == obj1):
            self.type = 'tri'
            self.text = "At point, find trisector ({} and {})".format(id(obj1), id(obj2))
            self.doable = True


class GeometricCollection:
    """Represents a collection of geometric objects."""

    def __init__(self):
        """Initializes the data."""
        self.population = []
        self.tasks = []
        self.data = []

        print("(Initializing GC = {})".format(id(self)))

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
        self.add_obj(x)
        self.data.append(np.rad2deg(x.theta))

        # Also add complement angle
        x = Angle(x.theta + x.alpha - np.pi, x.alpha, x.point) #hmmm... clearly needs a think
        self.add_obj(x)
        self.data.append(x.theta)

        return x

    def add_obj(self, new_obj):
        for obj in self.population:
            task = Task(obj, new_obj)
            if task.doable:
                self.tasks.append(task)

        self.population.append(new_obj)

    def get_tasks(self, task_type):
        new_tasks = []
        now_tasks = []
        for task in self.tasks:
            if task.type == task_type:
                now_tasks.append(task)
            else:
                new_tasks.append(task)
        self.tasks = new_tasks
        return now_tasks

    def connect_points(self):
        now_tasks = self.get_tasks('p2p')
        for task in now_tasks:
            p = task.obj1
            q = task.obj2
            dy = p.y-q.y
            dx = p.x-q.x

            self.line(p.x, p.y, np.arctan2(dy, dx), [p, q])

            self.data.append(np.sqrt(dx**2 + dy**2))

    def intersect_lines(self):
        now_tasks = self.get_tasks('lx')
        for task in now_tasks:
            l1 = task.obj1
            l2 = task.obj2

            p = set(l1.points).intersection(l2.points)

            if p:
                p = p.pop()

            else:

                # Check for parallel lines?
                if l1.theta == l2.theta:
                    if l1.theta == np.arctan2(l1.y-l2.y, l1.x-l2.x):
                        print("Lines are equal!! ({} and {})".format(id(l1), id(l2)))
                    else:
                        print("Lines are parallel!! ({} and {})".format(id(l1), id(l2)))

                c1 = np.tan(l1.theta)
                c2 = np.tan(l2.theta)

                x = (l2.y - l1.y - c2 * l2.x + c1 * l1.x) / (c1 - c2)
                y = (c1*l2.y - c2*l1.y - c1*c2 * l2.x + c1*c2 * l1.x) / (c1 - c2)

                p = self.point(x, y, [l1, l2])

            # Now create an angle from this intersection
            self.angle(l1.theta, l2.theta, p)

    def trisect_angles(self):
        now_tasks = self.get_tasks('tri')
        for task in now_tasks:
            p = task.obj1
            angle = task.obj2

            self.line(p.x, p.y, angle.alpha + angle.theta/3., [p])
            self.line(p.x, p.y, angle.alpha + angle.theta*2./3., [p])

    def what_tasks(self):
        """Prints the current population."""
        print("We have the following {:d} tasks outstanding:\n".format(len(self.tasks)))
        for task in self.tasks:
            print("{}\n".format(task.text))

    def show_data(self):
        """Prints the current population."""
        print("{}\n".format(self.data))

    def plot_constructions(self):

        for obj in reversed(self.population):
            obj.plot()

        #pylab.title('Probability distribution of anharmonic oscillator with beta=' + str(beta))
        #pylab.xlabel('position')
        #pylab.ylabel('probability')
        #pylab.legend(['matrix squaring', 'path sampled'])
        pylab.xlim(-1., 2.)
        pylab.ylim(-1., 2.)
        pylab.show()
        pylab.axis('scaled')


class Geometric:
    """Represents an abstract geometric notion, with a name."""

    def __init__(self, name):
        """Initializes the data."""
        self.name = name
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
        self.lines = lines
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
        self.points = points

    def plot(self):
        pylab.plot(
            [self.x + edge * np.cos(self.theta) for edge in [-2, 2]],
            [self.y + edge * np.sin(self.theta) for edge in [-2, 2]],
            linewidth=1, color='g')


class Angle(Geometric):
    """Represents an angle theta (inclined at angle alpha) at a point."""

    def __init__(self, angle1, angle2, point):
        Geometric.__init__(self, 'Angle')

        # TODO want the angle to point into the triangle if possible
        theta = angle1-angle2
        alpha = angle2

        self.theta = theta
        self.alpha = alpha
        self.point = point

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
triangle.connect_points()
triangle.intersect_lines()
triangle.trisect_angles()
triangle.intersect_lines()
triangle.connect_points()
# triangle.what_tasks()
triangle.show_data()
print min([abs(i-60) for i in triangle.data])
#triangle.plot_constructions()