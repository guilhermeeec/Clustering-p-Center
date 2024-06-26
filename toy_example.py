import numpy as np

from visualize_helper import plot_points
from pulp_solver import solve_pcenter_pulp
from kemans_solver import solve_kmeans

points = np.array([[1,1,0],[4,-1,1],[2,5,0],[5,4,1],[1,2,0],
                   [3,-1,1],[2,2,0],[2,3,0],[4.5,1,1],[3.5,0,1]])


cluster_colors = {0:'r',1:'g'}
plot_points(points, cluster_colors_dict=cluster_colors)
input("Press any key to continue...")

###### p-Center ######

# All points are candidates to be facilities
p=2
points, clusters_indexs = solve_pcenter_pulp(points, p)

cluster_colors = {clusters_indexs[0]:'r',clusters_indexs[1]:'g'}
plot_points(points,cluster_colors_dict=cluster_colors)
input("Press any key to continue...")

###### K-Means ######

p=2
points, clusters_centers = solve_kmeans(points, p)

cluster_colors = {0:'r',1:'g'}
plot_points(points,cluster_colors_dict=cluster_colors,centers=clusters_centers)
input("Press any key to continue...")
