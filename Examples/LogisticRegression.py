from Algorithms.SimpleGradient import SimpleGradient
from Regression.LogisticRegression import LogisticRegression
import numpy as np

from Regression.CostFunctions import maximum_like_hood, d_maximum_like_hood

# load data id,age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active,cardio
data = np.loadtxt('resources/cardio_train.csv', delimiter=',')


# split
xs, ys = np.c_[data[:30000, 3:5]], np.c_[data[:30000, 12]]
test_xs, test_ys = np.c_[data[6000:, 3:5]], np.c_[data[6000:, 12]]

# Optimize
gradient = SimpleGradient(cost_fn=maximum_like_hood, cost_d_fn=d_maximum_like_hood)

# We get our parameters
lr = LogisticRegression(optimizer=gradient,xs=xs,ys=ys)

thetas, j_vals = lr.run(steps=900000, alpha=8)

lr.plot()

lr.test(test_xs=test_xs,test_ys=test_ys)

# Predict
# lr.predict()
