from scipy.spatial.distance import cdist
import numpy as np

class Locations:

    def __init__(self) -> None:
        self.points = None

    def x_array(self):
        pass

    def y_array(self):
        pass

    def get_distance_matrix(self):
        return cdist(self.points, self.points, metric='euclidean')



class Points(Locations):

    def __init__(self):
        super().__init__()


class Grid(Locations):

    def __init__(self, width, height, nx, ny) -> None:
        super().__init__()

        self.dims = np.array([nx, ny])
        self.size = np.array([width, height])
       
        self._make_points()

    def x_array(self):
        return self.points[:, 0]
    
    def y_array(self):
        return self.points[:, 1]
        
    def _index(self, i, j):
        return i + self.dims[0]*j
    
    def _make_points(self):
        self.points = np.zeros(shape=(self.dims[0]*self.dims[1], 2), dtype=np.float64)
        sxy = self.size / (self.dims - 1)
        ori = - self.size / 2
        pos = np.zeros(shape=ori.shape)
        for i in range(self.dims[0]):
            pos[0] = i*sxy[0] + ori[0]
            for j in range(self.dims[1]):
                pos[1] = j*sxy[1] + ori[1]
                self.points[self._index(i, j), :] = pos
               
        

