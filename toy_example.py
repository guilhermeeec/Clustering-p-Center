import matplotlib.pyplot as plt
import numpy as np

from visualize_helper import plot_points
from pulp_solver import solve_pcenter_pulp

cluster_colors = {
    0:'r',
    1:'g'
}

points = np.array([[1,1,0],[4,-1,1],[2,5,0],[5,4,1],[1,2,0],
                   [3,-1,1],[2,2,0],[2,3,0],[4.5,1,1],[3.5,0,1]])
plot_points(points, cluster_colors)

# All points are candidates to be facilities
p=2
solve_pcenter_pulp(points, p)