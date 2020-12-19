import numpy as np


def linear_h(x, thetas):
    return np.dot(x, thetas.T)


def logistic_h(x, thetas):
    h = linear_h(x, thetas)

    return 1 / (1 + np.exp(-h))


def mse(xs, ys, thetas):
    h = linear_h
    m = np.size(xs, 1)
    b = sum(np.square((h(xs, thetas)) - ys))
    return (2 / m) * b


def d_mse(xs, ys, thetas):
    h = linear_h(xs, thetas)
    n = len(ys)

    gradient = np.dot(xs.T, (h - ys))

    return gradient / n


def maximum_like_hood(xs, ys, thetas):
    n = xs.shape[0]

    h = logistic_h(xs, thetas)

    positive = np.log(h + 1e-300) * ys

    negative = np.log(1 - h + 1e-300) * (1 - ys)

    cost = positive + negative

    return cost.sum() / -n


def d_maximum_like_hood(xs, ys, thetas):
    h = logistic_h(xs, thetas)
    # n = xs.shape[0]

    gradient = np.dot(xs.T, (h - ys))
    # TODO shouldnt n be removed?
    return gradient
