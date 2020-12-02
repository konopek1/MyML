import numpy as np

from Utils.Matrix import add_ones_column


def linear_h(x, thetas):
    return x @ thetas.T


def logistic_h(x, thetas):
    return 1 / (1 + np.exp(-linear_h(x, thetas)))


def mse(xs, ys, thetas):
    h = linear_h
    square = np.vectorize(np.square)
    m = np.size(xs, 1)
    b = sum(square((h(xs, thetas)) - ys))
    return (2 / m) * b


def d_mse(xs, ys, thetas, j):
    h = linear_h
    n = len(ys)
    return (1 / n) * sum((h(xs, thetas) - ys) * np.c_[xs[:, j]])


def maximum_like_hood(xs, ys, thetas):
    h = logistic_h
    n = len(ys)
    positive = np.log(h(xs, thetas)) * ys
    negative = np.log(1 - h(xs, thetas)) * (1 - ys)
    return (-1 / n) * sum(positive + negative)


def d_maximum_like_hood(xs, ys, thetas, j):
    h = logistic_h
    n = len(ys)
    return (1 / n) * sum((h(xs, thetas) - ys) * np.c_[xs[:, j]])


def with_regularization(func, kind, delta):
    """
    kind - For lasso regression 2 for ridge regression 1"""

    def wrapped(*args, **kwargs):
        return func(*args, **kwargs) + regularization(kind, delta)[1](args[2])

    return wrapped


def with_regularization_d(func, kind, delta):
    """
    For derevatives
    kind - For lasso regression 2 for ridge regression 1"""

    def wrapped(*args, **kwargs):
        return func(*args, **kwargs) + regularization(kind, delta)[0](args[2])

    return wrapped


def regularization(kind, delta):
    if kind is None:
        return lambda x: 0, lambda x: 0
    elif kind == 2:
        square = np.vectorize(np.square)
        return lambda thetas: 2 * delta * sum(sum(thetas[:, 1:])), lambda thetas: 2 * delta * sum(
            sum(square(thetas[:, 1:])))
