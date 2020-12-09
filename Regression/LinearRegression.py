from Algorithms.Optimizer import Optimizer
from PreProcessing.Utils import normalize
from Regression.CostFunctions import linear_h
from Utils.Matrix import add_ones_column
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
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
        return linear_h(xs, self.thetas).item()
