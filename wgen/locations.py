from scipy.spatial.distance import cdist
import numpy as np
from wgen.parameters import LiDARSimulationParameters

class Locations:

    @staticmethod
    def create(params: LiDARSimulationParameters):
        return Grid(params.GridWidth, params.GridHeight, int(params.GridLength), int(params.GridLength))

    def __init__(self) -> None:
        self.points = None

    def __len__(self):
        return self.points.shape[0]
    
    def point(self, index):
        return self.points[index]

    def x_array(self):
        pass

    def y_array(self):
        pass

    def get_distance_matrix(self):
        return cdist(self.points, self.points, metric='euclidean')
    


class Points(Locations):

    def __init__(self):
        super().__init__()


    def AddPoints(self,Y,Z):
        # Cette fonction permet d'ajouter des points au champ. Cette fonction est majeure, puisqu'elle permet 
        # de definir les points pour lequels le vent va etre genere.
        #TEST de l'existence du point qu'on souhaite ajouter.
        ii=0
        flag=0
        while ii<len(self.Points):
            if ((float(Y)-self.Points[ii][0])**2+(float(Z)-self.Points[ii][1])**2)**(1./2)<=0.01:
                #print('Point (%s,%s) existing @ indice %s' %(Y,Z,ii))
                flag=1
                break
            else:
                ii+=1
        # SI ce point n'est pas detecte, alors on ajout le point en fin de liste
        if not flag==1:
            self.Points.append([float(Y),float(Z)])
            self.Wind.append(LiDARWind(self.params))
            self.WindValuesInitialized=0
        return ii #renvoi de l'indice, pour la connaissance dans le champ de vent du point correspondant
        #print(self.Points)
        


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
               
        

