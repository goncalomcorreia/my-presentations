# most code here courtesy of Thomas Boggs
# https://gist.github.com/tboggs/8778945

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])
_midpoints = [(_corners[(i + 1) % 3] + _corners[(i + 2) % 3]) / 2.0
              for i in range(3)]


def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.

    Arguments:
        `xy`: A length-2 sequence containing the x and y value.
    '''
    s = [(_corners[i] - _midpoints[i]).dot(xy - _midpoints[i]) / 0.75
         for i in range(3)]

    return np.clip(s, tol, 1.0 - tol)


def bc2xy(P):
    return np.dot(P, _corners)


def draw_contours(f, nlevels=20, subdiv=8, **kwargs):
    '''Draws pdf contours over an equilateral triangle (2-simplex).
    Arguments:
        `dist`: A distribution instance with a `pdf` method.
        `border` (bool): If True, the simplex border is drawn.
        `nlevels` (int): Number of contours to draw.
        `subdiv` (int): Number of recursive mesh subdivisions to create.
        kwargs: Keyword args passed on to `plt.triplot`.
    '''
    from matplotlib import ticker, cm
    import math

    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [f(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
    return plt.tricontour(trimesh, pvals, nlevels, **kwargs)

def draw(f, triangle=True, corners=True, color='k', **kwargs):
    # draw the triangle

    if triangle:
        plt.triplot(_triangle, color=color)
    # label the corners

    if corners:
        for i, corner in enumerate(_corners):
            p1, p2, p3 = xy2bc(corner).round(1)
            label = "({:1.0f}, {:1.0f}, {:1.0f})".format(p1, p2, p3)
            print(label)
            va = 'top'
            sgn = -1
            if p3 == 1:
                va = 'bottom'
                sgn = +1.2

            plt.annotate(label, xy=corner,
                        xytext=(0, sgn * 5),
                        textcoords='offset points',
                        horizontalalignment='center',
                        verticalalignment=va)

    print("Value at center:", f(np.ones(3) / 3))
    eps = 1e-20
    print("Value on face:", f(np.array([0.5, 0.5 - eps, 0.0 + eps])))

    cs = draw_contours(f, subdiv=8,
                       cmap=plt.cm.plasma_r,
                       **kwargs)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, _corners[-1][-1] + 0.1)
    plt.axis('off')
    return cs


def scatter(points, *args, **kwargs):
    points = np.atleast_2d(points)
    points_xy = bc2xy(points)
    plt.scatter(points_xy[:, 0], points_xy[:, 1], *args, **kwargs)

