import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return np.maximum(0, x)


def hat(x, points):
    if not isinstance(x, (int, float)):
        return np.array([hat(xi, points) for xi in x]).flatten()
    x0, x1, x2 = points
    assert x0 < x1 < x2
    
    A = np.array([[1./(x1-x0), -1./(x2-x1)]]).T
    b = np.array([[-x0/(x1-x0), x2/(x2-x1)]]).T

    y1 = A.dot(x) + b
    y1 = relu(y1)
    
    A = np.array([[1, -1], [-1, 1]])
    b = np.zeros((2, 1))
    y2 = A.dot(y1) + b
    y2 = relu(y2)

    A = np.array([[1, 1]])
    b = np.array([[0]])
    y3 = A.dot(y2) + b

    y = np.vstack([y1, y3])
    A = np.array([[0.5, 0.5, -0.5]])

    return A.dot(y)


def p1(x, c, mesh):
    y = np.zeros_like(x)
    for i, ci in enumerate(c):
        y += ci*hat(x, mesh[i:i+3])
    return y

x = np.linspace(-1, 1, 1001)

mesh = np.linspace(-1, 1, 101)
coef = np.sin(np.pi*mesh[1:-1])

plt.figure()
plt.plot(x, p1(x, coef, mesh))
plt.show()
