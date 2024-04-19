from sklearn.datasets import make_blobs
import numpy as np
from visualize_helper import *
from pulp_solver import *

def generate_clusters(n_samples,centers,stdevs):
    assert(centers.shape[0] == stdevs.shape[0])
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=centers,
        cluster_std=stdevs,
        random_state=42)
    points = np.zeros((n_samples,3))
    points[:,0:2] = X
    points[:,2] = y.T
    return points
    
if __name__ == "__main__":
    centers = np.array([[-2,2],[2,2]])
    stdevs = np.array([2,2])
    points = generate_clusters(20,centers,stdevs)

    cluster_colors = {0:'r',1:'g'}
    plot_points(points,cluster_colors_dict=cluster_colors,centers=centers,verbose=True)
    input("Press any key to continue...")
    
    points, clusters_indexes = solve_pcenter_pulp(points, 2)
    cluster_colors = {clusters_indexes[0]:'r',clusters_indexes[1]:'g'}
    plot_points(points,cluster_colors_dict=cluster_colors)
    input("Press any key to continue...")