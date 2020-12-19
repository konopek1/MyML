import numpy as np

map_zeros_to_almost_zeros = np.vectorize(lambda x: 5e-324 if x == 0 else x)


class StdScaler:
    def __init__(self):
        self.std = None
        self.mean = None

    def fit(self, x):
        self.std = np.std(x, axis=0)
        self.mean = np.mean(x, axis=0)

    def transform(self, x):
        if self.std is None or self.mean is None:
            raise Exception("You should use .fit(x) first!! You noob")
        else:
            return (x - np.mean(x, axis=0)) / map_zeros_to_almost_zeros(np.std(x, axis=0))
