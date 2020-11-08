import numpy as np


def add_ones_column(xs: np.ndarray) -> np.ndarray:
    ones_col = np.ones((np.size(xs), 1))
    return np.concatenate((ones_col, xs), axis=1)
