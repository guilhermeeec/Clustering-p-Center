import matplotlib.pyplot as plt
import numpy as np

'''
Input: 
points: numpy array with [point1, point2, ...], where each point is [x,y,cluster_id]
cluster_colors_dict dictionary {cluster_id:color}
'''
def plot_points(points,cluster_colors_dict):
    plt.ion()

    clusters = np.unique(points[:, 2])
    for cluster in clusters:
        cluster_points = points[points[:, 2] == cluster]
        color = cluster_colors_dict.get(cluster, 'blue')  # Default to blue if cluster color is not specified
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {cluster}')
        
    for i, (x, y, _) in enumerate(points):
        plt.text(x, y, str(i), fontsize=12, ha='left', va='bottom')
        

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points Plot')
    plt.legend()
    plt.grid(True)

    plt.show()
    #plt.draw()
    plt.pause(0.001)