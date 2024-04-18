
import numpy as np

class Point:
    def __init__(self,x,y,cluster_name=None) -> None:
        self.x = x
        self.y = y
        self.coordinates = np.array([x,y])
        self.cluster_name = cluster_name
