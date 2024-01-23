import numpy as np
from yawisi.parameters import LiDARSimulationParameters
from yawisi.spectrum import LiDARSpectrum
from yawisi.locations import Locations, Grid
from yawisi.kernels import CoherenceKernel
from yawisi.wind import LiDARWind
from tqdm import tqdm

class LiDARWindField:
    """
    cette classe permet de definir un champ de vent contenant un certain nombre de points,
    et permet de generer le vecteur de vent
    """
    def __init__(self, params: LiDARSimulationParameters):
        self.params: LiDARSimulationParameters = params #Def des parametres de simulation pour le Wind Field
        self.coherence_kernel = CoherenceKernel()
        self.locations: Locations = Locations.create("grid", 
                                                  width=self.params.grid_width, 
                                                  height=self.params.grid_height, 
                                                  nx=self.params.grid_length, 
                                                  ny=self.params.grid_length
                                                  ) 
        self.wind=[]   #Objets vent contenus dans le champ

    def get_umean(self):
        assert isinstance(self.locations, Grid), " can only generate for a grid"
        n_points = len(self.locations)
        # center of the grid is the location in the middle of the list
        pt = self.locations.points[n_points // 2, :]
        print(pt)
        wind = self.wind[n_points//2]
        return wind.wind_values[:, 0].mean()

    def get_uvwt(self):
        """format (3 x ny x nz x nt)"""
        assert isinstance(self.locations, Grid), " can only generate for a grid"

        grid: Grid = self.locations 

        ny, nz = grid.dims[0], grid.dims[1]
        nt = self.params.n_samples
        ts = np.empty((3, ny, nz, nt))
        for idx, w in enumerate(self.wind):
            i, j = grid.coords(idx)
            ts[:, i, j, :] = np.transpose(w.wind_values)

        return ts

     
    @property
    def is_initialized(self) -> bool:
        return len(self.wind) > 0
    
    def get_coherence_function(self):
        N=self.params.n_samples
        Ts=self.params.sample_time

        freq = np.arange(0, 1/Ts, 1/(Ts*N))
        coherence_function = np.zeros(shape=(N, ))
        
        coherence_function[:N//2] = self.coherence_kernel(freq[:N//2])
        coherence_function = np.pad(coherence_function[:N//2], [0, N//2], mode='reflect')
        return freq, coherence_function
    
    def _get_coherence_matrix(self, factor, distance_matrix):
        return np.exp(-factor*distance_matrix)
    
    def compute(self):

        N = self.params.n_samples
        n_points = len(self.locations)

        spectrum = LiDARSpectrum(self.params) #Spectre du signal de vent
      

        #Definition des transformation de Fourier des seeds du vent en chaque point
        fft_seed = np.zeros(shape =(n_points, N, 3), dtype=np.complex64)
        for i_pt in range(n_points):
            fft_seed[i_pt, :, :] = LiDARWind.get_initial_fftseed(N)
            
            #Multiplication par la matrice de coherence
        
        distance_matrix = self.locations.get_distance_matrix() # store a distance matrix 
        _, coherence_function = self.get_coherence_function()
        for i in tqdm(range(N)):
            coherence_matrix = self._get_coherence_matrix(coherence_function[i], distance_matrix)
            L = np.linalg.cholesky(coherence_matrix)
            fft_seed[:, i, :] = np.dot(L, fft_seed[:, i, :])  

      
        for i_pt in range(n_points):
            pt = self.locations.points[i_pt]
            wind = LiDARWind(self.params)
            wind.wind_mean =  np.array(
              [
                 self.params.wind_mean*((self.params.reference_height+pt[1])/self.params.reference_height)**(self.params.vertical_shear),
                 0,
                 0   
              ])
            wind.compute(fft_seed=fft_seed[i_pt, :, :], spectrum=spectrum)
            self.wind.append(wind)