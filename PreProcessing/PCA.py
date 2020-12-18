import numpy as np


def PCA(input: np.ndarray, dims) -> np.ndarray:
    cor = COR(input)
    eigen_vals, eigen_vecs = np.linalg.eig(cor)
    eigen_vecs = sort_eigen(eigen_vals, eigen_vecs, dims=dims)

    return input @ eigen_vecs.T


def sort_eigen(eigen_vals, eigen_vecs, dims):
    indexed_vals = enumerate(eigen_vals)
    sorted_vals = sorted(indexed_vals, key=lambda val: val[1], reverse=True)
    choosen_indexes = list(map(lambda x: x[0], sorted_vals))[0:dims]

    return eigen_vecs[choosen_indexes, :]


def COR(input: np.ndarray) -> np.ndarray:
    rows, cols = np.shape(input)
    mean = np.mean(input, 0)
    return np.dot(input.T, (input - mean)) / cols
