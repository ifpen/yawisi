from scipy.spatial.distance import cdist
import numpy as np
from yawisi.parameters import SimulationParameters

class Locations:

    @staticmethod
    def create(kind: str, **kwargs):
        if kind == "grid":
            return Grid(**kwargs)
        elif kind == "points":
            return Points()
        else:
            raise ValueError(f"kind {kind} is not grid/points")

    def __init__(self) -> None:
        self.points = None

    def __len__(self):
        return self.points.shape[0]
    
    def point(self, index):
        return self.points[index]

    def y_array(self):
        return self.points[:, 0]
    
    def z_array(self):
        return self.points[:, 1]
    
    def get_distance_matrix(self):
        return cdist(self.points, self.points, metric='euclidean')
    


class Points(Locations):

    def __init__(self):
        super().__init__()


    def add_points(self, pts: list):
        self.points = np.array(pts)
        
    

class Grid(Locations):

    def __init__(self, width, height, nx, ny) -> None:
        super().__init__()

        self.dims = np.array([nx, ny])
        self.size = np.array([width, height])
       
        self._make_points()
        
    def _index(self, i, j):
        return i + self.dims[0]*j
    
    def coords(self, idx):
        return idx % self.dims[0], idx // self.dims[0] 
    
    def assign(self, y, z):
        """create points array for to list of coordinates"""
        assert y.shape[0] == self.dims[0], "not the right dimension in y direction"
        assert z.shape[0] == self.dims[1], "not the right dimension in z direction"
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                self.points[self._index(i, j), 0] = y[i]
                self.points[self._index(i, j), 1] = z[i]
    
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


               
        

