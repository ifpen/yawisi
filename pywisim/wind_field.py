import numpy as np
from pywisim.parameters import LiDARSimulationParameters
from pywisim.spectrum import LiDARSpectrum
from pywisim.locations import Locations
from pywisim.kernels import CoherenceKernel
from pywisim.wind import LiDARWind
from tqdm import tqdm

class LiDARWindField:
    """
    cette classe permet de definir un champ de vent contenant un certain nombre de points,
    et permet de generer le vecteur de vent
    """
    def __init__(self, params: LiDARSimulationParameters):
        self.params: LiDARSimulationParameters = params #Def des parametres de simulation pour le Wind Field
        
        self.coherence_kernel = CoherenceKernel()
        self.Spectrum = LiDARSpectrum(params) #Spectre du signal de vent
       
        
        self.Points: Locations = Locations.create(params) #Points du champ de vent
        self.Wind=[]   #Objets vent contenus dans le champ
       
     
        self.WindValuesInitialized=0 # Flag pour l'initialisation des valeurs de vent
    
    def get_coherence_function(self):
        N=self.params.NSamples
        Ts=self.params.SampleTime

        freq = np.arange(0, 1/Ts, 1/(Ts*N))
        coherence_function = np.zeros(shape=(N, ))
        
        coherence_function[:N//2] = self.coherence_kernel(freq[:N//2])
        coherence_function = np.pad(coherence_function[:N//2], [0, N//2], mode='reflect')
        return freq, coherence_function
    

    def _compute_fft_seed(self):

        N = self.params.NSamples
        n_points = len(self.Points)

        self.Spectrum.compute(Npts=N, )

        #Definition des transformation de Fourier des seeds du vent en chaque point
        fft_seed = np.zeros(shape =(n_points, N, 3), dtype=np.complex64)
        for i_pt in range(n_points):
            fft_seed[i_pt, :, :] = LiDARWind.get_initial_fftseed(N)
            
            #Multiplication par la matrice de coherence
        
        distance_matrix = self.Points.get_distance_matrix() # store a distance matrix 
        _, coherence_function = self.get_coherence_function()
        for i in tqdm(range(N)):
            coherence_matrix = np.exp(-coherence_function[i]*distance_matrix)
            L = np.linalg.cholesky(coherence_matrix)
            fft_seed[:, i, :] = np.dot(L, fft_seed[:, i, :])  

      
        for i_pt in range(n_points):
            pt = self.Points.points[i_pt]
            wind = LiDARWind(self.params)
            wind.wind_mean =  np.array(
              [
                 self.params.WindMean*((self.params.ReferenceHeight+pt[1])/self.params.ReferenceHeight)**(self.params.VerticalShearParameter),
                 0,
                 0   
              ])
            wind.compute(fft_seed=fft_seed[i_pt, :, :], lidar_spectrum=self.Spectrum)
            self.Wind.append(wind)
