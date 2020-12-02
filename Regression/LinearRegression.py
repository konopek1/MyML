from Algorithms.Optimizer import Optimizer
from Utils.Matrix import add_ones_column
import numpy as np
import matplotlib.pyplot as plt


class Regression:
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
        plt.subplot(2, 1, 1)
        plt.plot(self.j_values)
        plt.show()

    def describe(self):
        pass

    def predict(self, xs: np.ndarray):
        xs = add_ones_column(xs)
        return self.optimizer.h(xs, self.thetas)

    def test(self,test_xs,test_ys):
        n = len(test_ys)
        test_xs = add_ones_column(test_xs)
        acc = 0
        for i in range(n):
            h = self.optimizer.h(test_xs[i,:], self.thetas) > 0.5
            acc += h == test_ys[i]
        return acc / n


def mse(h, xs, ys, thetas):
    square = np.vectorize(np.square)
    m = np.size(xs, 1)
    b = sum(square((h(xs, thetas)) - ys))
    return (2 / m) * b


def d_mse(h, xs, ys, thetas, j):
    n = len(ys)
    return (1 / n) * sum((h(xs, thetas) - ys) * np.c_[xs[:, j]])


def maximum_like_hood(h, xs, ys, thetas):
    n = len(ys)
    positive = np.log(h(xs, thetas)) * ys
    negative = np.log(1 - h(xs, thetas)) * (1 - ys)
    return (-1 / n) * sum(positive + negative)


def d_maximum_like_hood(h, xs, ys, thetas, j):
    n = len(ys)
    return sum((h(xs, thetas) - ys) * np.c_[xs[:, j]])
