import numpy as np


class Optimizer:
    def __init__(self):
        self.ys = None
        self.xs = None
        self.h = None

    def run(self, steps: int, alpha: int, xs: np.ndarray, ys: np.ndarray, thetas=None):
        raise NotImplementedError("Abstract class")
