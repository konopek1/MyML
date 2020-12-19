import numpy as np

from Algorithms.SimpleGradient import SimpleGradient
from PreProcessing.StdScaler import StdScaler
from Regression.CostFunctions import maximum_like_hood, d_maximum_like_hood
from Regression.LogisticRegression import LogisticRegression

# load data id,age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active,cardio
data = np.loadtxt('resources/cardio_train.csv', delimiter=',')

# split
xs, ys = np.c_[data[:30000, 1:10]], np.c_[data[:30000, 12]]

test_xs, test_ys = np.c_[data[30000:, 1:10]], np.c_[data[30000:, 12]]

# standarize
std_scaler = StdScaler()
std_scaler.fit(xs)
xs = std_scaler.transform(xs)
test_xs = std_scaler.transform(test_xs)

# Optimize
gradient = SimpleGradient(cost_fn=maximum_like_hood, cost_d_fn=d_maximum_like_hood)

# We get our parameters
lr = LogisticRegression(optimizer=gradient, xs=xs, ys=ys)

thetas, j_vals = lr.run(steps=5000, alpha=0.00006)

lr.plot()

print(lr.test(test_xs=test_xs, test_ys=test_ys))

# Predict
# lr.predict()
