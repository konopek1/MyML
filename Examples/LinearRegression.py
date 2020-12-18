import numpy as np

from Algorithms.SimpleGradient import SimpleGradient
from Regression.CostFunctions import d_mse, mse, linear_h
from Regression.LinearRegression import LinearRegression
from Regression.Regularization import with_regularization_d, with_regularization

# load data
data = np.loadtxt('resources/ex1data1.txt', delimiter=',')

# split
xs, ys = np.c_[data[:, 0]], np.c_[data[:, 1]]

# Optimize
gradient = SimpleGradient(cost_fn=mse, cost_d_fn=d_mse)

# We get our parameters
thetas, j_values = gradient.run(steps=1500, alpha=0.006, xs=xs, ys=ys)

# Predict
linear_h([1, xs[12]], thetas).item()
print(ys[12])

# We will add regularization
r_mse = with_regularization(mse, 2, 2)
r_d_mse = with_regularization_d(d_mse, 2, 2)
optimizer = SimpleGradient(cost_fn=r_mse, cost_d_fn=r_d_mse)

# We can also wrap gradient in LinearRegression Class like this
lr = LinearRegression(optimizer=optimizer, xs=xs, ys=ys)

# Just like gradient.run
thetas, j_values = lr.run(steps=1500, alpha=0.01)

# But have some additional capibilities

# Plot
lr.plot()

# Predict
lr.predict(np.c_[xs[1]])
