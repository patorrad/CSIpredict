import numpy as np
import matplotlib.pylab as plt
import math

def deCasteljau(points, u, k = None, i = None, dim = None):
    """Return the evaluated point by a recursive deCasteljau call
    Keyword arguments aren't intended to be used, and only aid
    during recursion.

    Args:
    points -- list of list of floats, for the control point coordinates
              example: [[0.,0.], [7,4], [-5,3], [2.,0.]]
    u -- local coordinate on the curve: $u \in [0,1]$

    Keyword args:
    k -- first parameter of the bernstein polynomial
    i -- second parameter of the bernstein polynomial
    dim -- the dimension, deduced by the length of the first point
    """
    if k == None: # topmost call, k is supposed to be undefined
        # control variables are defined here, and passed down to recursions
        k = len(points)-1
        i = 0
        dim = len(points[0])

    # return the point if downmost level is reached
    if k == 0:
        return points[i]

    # standard arithmetic operators cannot do vector operations in python,
    # so we break up the formula
    a = deCasteljau(points, u, k = k-1, i = i, dim = dim)
    b = deCasteljau(points, u, k = k-1, i = i+1, dim = dim)
    result = []

    # finally, calculate the result
    for j in range(dim):
        result.append((1-u) * a[j] + u * b[j])

    return result

points = [[0.,0.], [2,1], [3,3]]

def plotPoints(b):
    x = [a[0] for a in b]
    y = [a[1] for a in b]
    plt.plot(x,y,'.')

curve = []

for i in np.linspace(0,1,100):
    curve.append(deCasteljau(points, i))

plotPoints(curve)
plt.show()