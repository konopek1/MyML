import numpy as np


def normalize(m: np.ndarray):
    _, n_features = np.shape(m)
    stds = []
    means = []

    for feature_index in range(n_features):
        std = np.std(m[:, feature_index])
        stds.append(std)
        mean = np.mean(m[:, feature_index])
        means.append(mean)
        norm = np.vectorize(lambda x: (x - mean) / std)
        m[:, feature_index] = norm(m[:, feature_index])
    return means, stds


def normalize_test(m, means, stds):
    """ Normalize by given means and stds
    genreated by normalize()"""
    _, n_features = np.shape(m)
    for feature_index in range(n_features):
        mean = means[feature_index]
        std = stds[feature_index]
        norm = np.vectorize(lambda x: (x - mean) / std)
        m[:, feature_index] = norm(m[:, feature_index])
