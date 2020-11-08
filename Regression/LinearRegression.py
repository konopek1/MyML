from Algorithms.Optimizer import Optimizer
from Utils.Matrix import add_ones_column
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, optimizer: Optimizer, xs, ys):
        self.xs = xs
        self.ys = ys
        self.optimizer = optimizer
        self.cost_fn = cost_fn
        self.thetas = None
        self.j_values = None

    def run(self, steps, alpha) -> (np.ndarray, np.ndarray):
        self.thetas, self.j_values = self.optimizer.run(steps, alpha, self.xs, self.ys)
        return self.thetas, self.j_values

    def plot(self):
        plt.subplot(2, 1, 1)
        plt.plot(self.j_values)
        plt.subplot(2, 1, 2)
        plt.plot(self.xs, self.ys, 'ro')
        plt.plot(self.xs, self.optimizer.h(add_ones_column(self.xs), self.thetas))
        plt.show()

    def describe(self):
        pass

    def predict(self, xs: np.ndarray):
        xs = add_ones_column(xs)
        return self.optimizer.h(xs, self.thetas)


def cost_fn(xs, ys, thetas):
    square = np.vectorize(np.square)
    m = np.size(xs, 1)
    b = sum(square((xs @ thetas.T) - ys))
    return (1 / (2 * m)) * b
