import matplotlib.pyplot as plt
import numpy as np

from Clustering.KMeans import Point, Centroid, KMean

points = [Point(size=(1, 2)) for _ in range(0, 100)]
centroids = [Centroid(size=(1, 2)) for _ in range(0, 10)]

kmean_result = KMean(centroids, points, 20).run()

for centroid in centroids:
    points_raw = np.array([point.pos for point in centroid.points])
    plt.scatter(points_raw[:, 0], points_raw[:, 1])

centroids_raw = np.array([centroid.pos for centroid in centroids])

plt.scatter(centroids_raw[:, 0], centroids_raw[:, 1], c='blue', marker='x')

plt.show()
