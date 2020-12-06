import numpy as np
from Algorithms.Optimizer import Optimizer
from Utils.Matrix import add_ones_column


class SimpleGradient(Optimizer):
    def __init__(self, cost_fn, cost_d_fn):
        super().__init__()
        self.cost_fn = cost_fn
        self.cost_d_fn = cost_d_fn

    def run(self, steps: int, alpha: float, xs: np.ndarray, ys: np.ndarray, thetas=None) -> (np.ndarray, np.ndarray):
        xs = add_ones_column(xs)
        n_features = np.size(xs, 1)
        j_values: np.ndarray = np.zeros((steps, 1))

        if thetas is None:
            thetas = np.zeros((1, n_features))

        for i in range(steps):
            grad = self.cost_d_fn(xs, ys, thetas).T

            thetas -= (alpha * grad)

            j_values[i] = self.cost_fn(xs, ys, thetas)

        return thetas, j_values
