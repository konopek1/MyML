import matplotlib.pyplot as plt
import numpy as np

from Clustering.KMeans import Point, KMean
from Clustering.SmartCentroids import smart_centroids

points = [Point(size=(1, 2)) for _ in range(0, 100)]
centroids = smart_centroids(points, 10)

centroids_raw = np.array([centroid.pos for centroid in centroids])

kmean_result = KMean(centroids, points, 200).run()

print(kmean_result)
centroids = kmean_result['centroids']

for centroid in centroids:
    # some clusters may be empty
    if centroid.points:
        points_raw = np.array([point.pos for point in centroid.points])
        plt.scatter(points_raw[:, 0], points_raw[:, 1])

centroids_raw = np.array([centroid.pos for centroid in centroids])

plt.scatter(centroids_raw[:, 0], centroids_raw[:, 1], c='blue', marker='x')

plt.show()
