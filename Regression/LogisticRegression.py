import matplotlib.pyplot as plt
import numpy as np

from Algorithms.Optimizer import Optimizer
from Regression.CostFunctions import logistic_h
from Utils.Matrix import add_ones_column


class LogisticRegression:
    def __init__(self, optimizer: Optimizer = None, xs=None, ys=None):
        self.xs = xs
        self.ys = ys
        self.optimizer = optimizer
        self.thetas = None
        self.j_values = None

    def run(self, steps, alpha) -> (np.ndarray, np.ndarray):
        self.thetas, self.j_values = self.optimizer.run(steps, alpha, self.xs, self.ys)
        return self.thetas, self.j_values

    def plot(self):
        plt.plot(self.j_values)
        plt.show()

    # TODO
    def describe(self):
        pass

    def predict(self, xs: np.ndarray):
        return logistic_h(self.thetas, xs.T).item()

    def test(self, test_xs, test_ys):
        n = len(test_ys)
        acc = 0
        test_xs = add_ones_column(test_xs)
        for i in range(n):
            h = self.predict(np.c_[test_xs[i, :]]) > 0.5
            acc += (h == test_ys[i].item())
        return acc / n
