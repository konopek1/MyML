from Algorithms.SimpleGradient import SimpleGradient
from Regression.LinearRegression import Regression, mse, d_mse, maximum_like_hood, d_maximum_like_hood
from PreProcessing.Utils import normalize, normalize_test
import numpy as np


# data = np.loadtxt('ex1data1.txt', delimiter=',')
# xs, ys = np.c_[data[:, 0]], np.c_[data[:, 1]]
#
linear_h = lambda x, t: x @ t.T
logistic_h = lambda x, t: 1 / (1 + np.exp(-linear_h(x, t)))
#
# linear_gradient = SimpleGradient(linear_h, cost_fn=mse, cost_d_fn=d_mse, kind_of_regularization=2, delta=5)
# linear_regression = Regression(linear_gradient, xs, ys)
#
# thetas, j_values = linear_regression.run(steps=1500, alpha=0.006)
# linear_regression.plot()

data = np.loadtxt('cardio_train1.csv', delimiter=',', skiprows=1)
train_data, test_data = data[:10000, :], data[600:, :]

train_xs, train_ys = train_data[:, :12], train_data[:, 12:]
means, stds = normalize(train_xs)

logistic_gradient = SimpleGradient(logistic_h, cost_fn=maximum_like_hood, cost_d_fn=d_maximum_like_hood,
                                   kind_of_regularization=2)
logistic_regression = Regression(logistic_gradient, train_xs, train_ys)

thetas, j_values = logistic_regression.run(steps=1000, alpha=0.001)

normalize_test(test_data[:,:12],means,stds)
logistic_regression.test(test_data[:,:12], test_data[:,12:])
logistic_regression.plot()

print(thetas)

