import numpy as np


class PcaScaler:
    def __init__(self):
        self.eigen_ves = None

    def fit(self, input, dims):
        cor = COV(input)
        eigen_vals, eigen_vecs = np.linalg.eigh(cor)
        self.eigen_vecs = sort_eigen(eigen_vals, eigen_vecs, dims=dims)

        return (input @ eigen_vecs.T).astype('float64')

    def transform(self, input):
        return (input @ self.eigen_vecs.T).astype('float64')


def sort_eigen(eigen_vals, eigen_vecs, dims) -> np.ndarray:
    idx = np.argsort(eigen_vals)[::-1]

    return eigen_vecs[idx[:dims], :]


def COV(input: np.ndarray) -> np.ndarray:
    rows, cols = np.shape(input)
    mean = np.mean(input, 0)
    return np.dot(input.T, (input - mean)) / cols
