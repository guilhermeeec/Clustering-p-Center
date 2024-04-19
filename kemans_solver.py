import numpy as np
from sklearn.cluster import KMeans

def solve_kmeans(points, p):
    coordinates = np.delete(points,-1,axis=1)
    kmeans_solver = KMeans(n_clusters=p, random_state=0, n_init="auto").fit(coordinates)

    print(kmeans_solver.labels_)
    for point_index, label in enumerate(kmeans_solver.labels_):
        points[point_index][2] = label

    return points, kmeans_solver.cluster_centers_