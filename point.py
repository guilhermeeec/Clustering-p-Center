
import numpy as np

def distance(p1, p2):
    return np.linalg.norm(p1.coordinates-p2.coordinates)

class Point:
    def __init__(self, coordinates, coordinates_labels=None, cluster_index=None):
        self.coordinates = np.array(coordinates) 
        assert(len(coordinates.shape)==1)
        self.dimension = coordinates.shape[0] #line vector
        if coordinates_labels is not None:
            assert(self.dimension == len(coordinates_labels))
            self.coordinates_labels = coordinates_labels
        self.coordinates = coordinates
        self.cluster_index = cluster_index