from typing import Tuple

import numpy as np


class Point:
    def __init__(self, pos: np.ndarray = None, size: Tuple[int, int] = None):
        if pos is None:
            if size is None:
                raise ValueError()
            else:
                self.pos = np.random.random(size)[0]
        else:
            self.pos = pos

    def distance(self, target):
        return np.linalg.norm(self.pos - target.pos)


class Centroid(Point):
    def __init__(self, pos: np.ndarray = None, size: Tuple[int, int] = None):
        super().__init__(pos=pos, size=size)
        self.points = []
        self.old_pos = None

    def clear(self):
        self.points = []

    def add_point(self, point: Point):
        self.points.append(point)

    def move(self):
        self.old_pos = self.pos
        np_points = np.array(list(map(lambda point: point.pos, self.points)))
        if np_points.size != 0:
            self.pos = [np.average(np_points[:, column]) for column in range(np.shape(np_points)[1])]

    def has_not_changed(self):
        return np.array_equal(self.pos, self.old_pos)

    @classmethod
    def from_point(cls, point: Point):
        new_centroid = cls(pos=point.pos)
        return new_centroid
