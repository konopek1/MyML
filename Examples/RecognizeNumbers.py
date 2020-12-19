import numpy as np

from Algorithms.OneVsAll import Class, OneVsAll
from Algorithms.SimpleGradient import SimpleGradient
from PreProcessing.PcaScaler import PcaScaler
from PreProcessing.StdScaler import StdScaler
from Regression.CostFunctions import maximum_like_hood, d_maximum_like_hood
from Regression.LogisticRegression import LogisticRegression

data = np.loadtxt('resources/mnist_train.csv', delimiter=',')
data_test = np.loadtxt('resources/mnist_test.csv', delimiter=',')

test_ys, test_xs = np.c_[data_test[:, 0]], data_test[:, 1:]
ys, xs = np.c_[data[:, 0]], data[:, 1:]

# Standardize
std_scaler = StdScaler()
std_scaler.fit(xs)
xs = std_scaler.transform(xs)
test_xs = std_scaler.transform(test_xs)

# PCA - reductions of dims
# reduce 50% of parameters
pca_scaler = PcaScaler()
pca_scaler.fit(xs, 350)

xs = pca_scaler.transform(xs)
test_xs = pca_scaler.transform(test_xs)

classes = [Class(i) for i in range(10)]

gradient = SimpleGradient(cost_fn=maximum_like_hood, cost_d_fn=d_maximum_like_hood)

lr = LogisticRegression(optimizer=gradient, xs=xs, ys=ys)

# First argument is class with run method which returns thetas, j_vals
lr = OneVsAll(lr, classes=classes)

# It runs LogisticRegression.run for each class
results = lr.run(steps=100, alpha=0.0006)

print(lr.test(test_xs, test_ys))
