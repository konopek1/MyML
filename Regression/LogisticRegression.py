from Algorithms.Optimizer import Optimizer
from Regression.CostFunctions import logistic_h
from Utils.Matrix import add_ones_column
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, optimizer: Optimizer, xs, ys):
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
        xs = add_ones_column(xs)
        return logistic_h(xs, self.thetas)

    def test(self, test_xs, test_ys):
        n = len(test_ys)
        test_xs = add_ones_column(test_xs)
        acc = 0
        for i in range(n):
            h = self.optimizer.h(test_xs[i, :], self.thetas) > 0.5
            acc += h == test_ys[i]
        return acc / n
