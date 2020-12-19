from copy import copy

import numpy as np

from Regression.CostFunctions import logistic_h
from Utils.Matrix import add_ones_column


class Class:
    def __init__(self, value):
        """Label should be vector"""
        self.value = value

    def map_to_class(self, labels):
        binary = np.vectorize(lambda x: 1 if x == self.value else 0)
        return binary(labels)

    def __str__(self):
        return f"{self.value}"


class OneVsAll:
    def __init__(self, binaryAlgorithm, classes: [Class]):
        self.runner = binaryAlgorithm
        self.classes = classes
        self.results = dict.fromkeys(classes)

    def run(self, *args, **kwargs):
        for k, v in self.results.items():
            copy_runner = copy(self.runner)
            copy_runner.ys = k.map_to_class(copy_runner.ys)
            thetas, jvals = copy_runner.run(*args, **kwargs)

            self.results[k] = thetas

        return self.results

    def predict(self, xs):
        def partial(xs, thetas):
            xs = add_ones_column(xs.T)
            return logistic_h(xs, thetas).item()

        rv = dict.fromkeys(self.classes)

        for k, thetas in self.results.items():
            rv[k] = partial(xs, thetas)

        return sorted(rv.items(), key=lambda x: x[1], reverse=True)[0]

    def test(self, test_xs, test_ys):
        n = len(test_ys)
        acc = 0
        for i in range(n):
            predicted = self.predict(np.c_[test_xs[i]])[0].value
            acc += (predicted == test_ys[i])
        return acc / n
