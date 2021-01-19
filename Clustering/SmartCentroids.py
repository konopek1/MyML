from typing import List

import numpy as np

from Clustering.Centroid import Point, Centroid


def smart_centroids(points: List[Point], n_centroids: int):
    centroids = []

    rand = np.random.randint(0, np.shape(points)[0])
    next_point = points[rand]

    for i in range(n_centroids):
        new_centroid = Centroid.from_point(next_point)
        centroids.append(new_centroid)

        current_distance = next_point.distance(new_centroid)

        for point in points:
            distance = new_centroid.distance(point)
            if distance > current_distance:
                next_point = point
                current_distance = distance

    return centroids
