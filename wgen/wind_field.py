import numpy as np
from wgen.parameters import LiDARSimulationParameters
from wgen.spectrum import LiDARSpectrum
from wgen.locations import Locations
from wgen.kernels import CoherenceKernel

class LiDARWindField:
    """
    cette classe permet de definir un champ de vent contenant un certain nombre de points,
    et permet de generer le vecteur de vent
    """
    def __init__(self, params: LiDARSimulationParameters):
        self.coherence_kernel = CoherenceKernel()
        self.params: LiDARSimulationParameters = params #Def des parametres de simulation pour le Wind Field
        self.Points: Locations = Locations.create(params) #Points du champ de vent
        self.Spectrum = LiDARSpectrum(params) #Spectre du signal de vent
        self.Wind=[]   #Objets vent contenus dans le champ
       
        self.Grid=[] # Initialisation des points de la grille
      
        self.WindValuesInitialized=0 # Flag pour l'initialisation des valeurs de vent
    
    def get_coherence_function(self):
        N=self.params.NSamples
        Ts=self.params.SampleTime

        freq = np.arange(0, 1/Ts, 1/(Ts*N))
        coherence_function = np.zeros(shape=(N, ))
        
        coherence_function[:N//2] = self.coherence_kernel(freq[:N//2])
        coherence_function = np.pad(coherence_function[:N//2], [0, N//2], mode='reflect')
        return freq, coherence_function
    
    
    def GetCoherenceMatrix(self,CoherenceParameter):
        #Cette fonction permet de recuperer la matrice de coherence
        i=0
        Matrix=array([[float(0)]*len(self.Points)]*len(self.Points))
        #print('Coherence matrice for a coherence of %s' % CoherenceParameter)
        while i<len(self.Points):
            j=0
            while j<len(self.Points):
                Matrix[i][j]=exp(-CoherenceParameter*((float(self.Points[i][0])-float(self.Points[j][0]))**2+(float(self.Points[i][1])-float(self.Points[j][1]))**2)**(1./2))
                j+=1
            #print(Matrix[i,:])
            i+=1
        return Matrix