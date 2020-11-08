from Algorithms.SimpleGradient import SimpleGradient
from Regression.LinearRegression import LinearRegression, cost_fn
import numpy as np


data = np.loadtxt('ex1data1.txt', delimiter=',')
xs, ys = np.c_[data[:, 0]], np.c_[data[:, 1]]

gradient = SimpleGradient(lambda xs, thetas: xs @ thetas.T,cost_function=cost_fn, kind_of_regularization=2, delta=5)
linear_regression = LinearRegression(gradient,xs,ys)

thetas, j_values = linear_regression.run(steps=1500, alpha=0.006)

linear_regression.plot()

print(thetas)
