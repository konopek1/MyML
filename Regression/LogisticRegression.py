from Algorithms.Optimizer import Optimizer
from PreProcessing.Utils import normalize, normalize_by
from Regression.CostFunctions import logistic_h
from Utils.Matrix import add_ones_column
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, optimizer: Optimizer, xs, ys):
        self.xs, self.norms = normalize(xs)
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
        xs = normalize_by(xs.T,norms=self.norms)

        xs = add_ones_column(xs)

        return logistic_h(xs, self.thetas).item()

    def test(self, test_xs, test_ys):
        n = len(test_ys)
        acc = 0
        for i in range(n):
            h = self.predict(np.c_[test_xs[i, :]]) > 0.5
            acc += (h == test_ys[i])
        return acc / n
