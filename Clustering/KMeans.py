from typing import List

from Clustering.Centroid import Centroid, Point


class KMean:
    def __init__(self, centroids: List[Centroid], points: List[Point], max_steps: int):
        self.centroids = centroids
        self.points = points
        self.max_step = max_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        [centroid.clear() for centroid in self.centroids]

        for point in self.points:
            centroid_distance = {centroid: centroid.distance(point) for centroid in self.centroids}

            sorted_centroid_distance = sorted(centroid_distance.items(), key=lambda item: item[1])

            closest_centroid = sorted_centroid_distance[0][0]

            closest_centroid.add_point(point)

        [centroid.move() for centroid in self.centroids]

    def max_iter_exceeded(self):
        if self.max_step == self.current_step:
            raise Success("Maximum iter exceeded")

    def optimum_reached(self):
        if all(map(lambda x: x.has_not_changed, self.centroids)):
            raise Success("Optimum reached")

    def run(self):
        try:
            while True:
                self.step()
                self.max_iter_exceeded()
                self.optimum_reached()
        except Success as success:
            print(f"KMeans finished, cause : {success}")
            return self.centroids


class Success(Exception):
    def __init__(self, message):
        super().__init__(message)
