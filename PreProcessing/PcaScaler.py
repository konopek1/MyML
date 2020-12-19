import numpy as np


class PcaScaler:
    def __init__(self):
        self.eigen_ves = None

    def fit(self, input, dims):
        cor = COR(input)
        eigen_vals, eigen_vecs = np.linalg.eigh(cor)
        self.eigen_vecs = sort_eigen(eigen_vals, eigen_vecs, dims=dims)

        return (input @ eigen_vecs.T).astype('float64')

    def transform(self, input):
        return (input @ self.eigen_vecs.T).astype('float64')


def sort_eigen(eigen_vals, eigen_vecs, dims) -> np.ndarray:
    indexed_vals = enumerate(eigen_vals)
    sorted_vals = sorted(indexed_vals, key=lambda val: val[1], reverse=True)
    choosen_indexes = list(map(lambda x: x[0], sorted_vals))[0:dims]

    return eigen_vecs[choosen_indexes, :]


def COR(input: np.ndarray) -> np.ndarray:
    rows, cols = np.shape(input)
    mean = np.mean(input, 0)
    return np.dot(input.T, (input - mean)) / cols
