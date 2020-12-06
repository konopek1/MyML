import numpy as np


def normalize(xs: np.ndarray):
    _, n_features = np.shape(xs)
    c_xs = np.copy(xs)
    norms = np.linalg.norm(c_xs,axis=0)

    for feauture_index in range(n_features):
        c_xs[:,feauture_index] /= norms[feauture_index]

    return c_xs,norms


def normalize_by(xs: np.ndarray, norms):
    _, n_features = np.shape(xs)
    c_xs = np.copy(xs)

    for feauture_index in range(n_features):
        c_xs[:,feauture_index] /= norms[feauture_index]

    return c_xs

