import numpy as np
from Algorithms.Optimizer import Optimizer
from Utils.Matrix import add_ones_column


class SimpleGradient(Optimizer):
    def __init__(self, h, cost_fn, cost_d_fn, kind_of_regularization=None, delta=1000):
        super().__init__()
        self.cost_fn = cost_fn
        self.cost_d_fn = cost_d_fn
        self.h = h
        self.regularization_d, self.regularization_fn = regularization(kind_of_regularization, delta)

    def run(self, steps: int, alpha: int, xs: np.ndarray, ys: np.ndarray, thetas=None) -> (np.ndarray, np.ndarray):
        xs = add_ones_column(xs)
        n_features = np.size(xs, 1)
        n = len(ys)
        j_values: np.ndarray = np.zeros((steps, 1))
        grad: np.ndarray = np.zeros((1, n_features))

        if thetas is None:
            thetas = np.zeros((1, n_features))

        for i in range(steps):

            for j in range(n_features):
                grad[0, j] = alpha * (self.cost_d_fn(self.h,xs,ys,thetas,j) + self.regularization_d(thetas)/n)

            thetas -= grad

            j_values[i] = self.cost_fn(self.h,xs, ys, thetas) + self.regularization_fn(thetas)

        return thetas, j_values


def regularization(kind, delta):
    if kind is None:
        return lambda x: 0, lambda x: 0
    elif kind == 2:
        square = np.vectorize(np.square)
        return lambda thetas: 2 * delta * sum(sum(thetas[:,1:])), lambda thetas: 2 * delta * sum(sum(square(thetas[:,1:])))
