import numpy as np


def with_regularization(func, kind, delta):
    """
    func - (xs,ys,thetas)
    kind - For lasso regression 2 for ridge regression 1"""

    def wrapped(*args, **kwargs):
        return func(*args, **kwargs) + regularization(kind, delta)[1](args[2]) / len(args[1])

    return wrapped


def with_regularization_d(func, kind, delta):
    """
    func - (xs,ys,thetas)
    For derevatives
    kind - For lasso regression 2 for ridge regression 1"""

    def wrapped(*args, **kwargs):
        return func(*args, **kwargs) + regularization(kind, delta)[0](args[2]) / len(args[1])

    return wrapped


def regularization(kind, delta):
    if kind is None:
        return lambda x: 0, lambda x: 0

    elif kind == 1:
        def f(thetas):
            return delta * sum(np.abs(thetas[:, 1:])).item()

        def d_f(thetas):
            return delta * sum(np.abs(thetas[:, 1:])).item()

        return d_f, f

    elif kind == 2:
        def f(thetas):
            return delta * sum(np.square(thetas[:, 1:])).item()

        def d_f(thetas):
            return 2 * delta * sum(thetas[:, 1:]).item()

        return d_f, f
