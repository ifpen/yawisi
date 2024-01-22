import numpy as np

from pywisim.parameters import LiDARSimulationParameters
from pywisim.spectrum import LiDARSpectrum


class LiDARWind:
    """
    Cette classe permet de definir un objet Vent, contenant une seed (pour chaque composante)
    Elle permet egalement de contenir les valeurs du vent, qui peuvent etre initialisee 
    a partir de la classe spectre, ou par la fonction du champ de vent.
    """

    @staticmethod
    def get_initial_fftseed(N):
        fft_seed = np.zeros(shape=(N,3), dtype=np.complex64)
        #fft avec mise à zero de la moyenne de la seed
        for i in range(fft_seed.shape[1]):
            seed = np.random.normal(size=(N,))
            fft_seed[:, i]= np.fft.fft(seed - np.mean(seed))
        return fft_seed
    
    def __init__(self,params: LiDARSimulationParameters):
        #initialisation des seeds a  0 et du vent a 0
        self.params = params
        self.wind_mean = np.array([self.params.WindMean, 0, 0])
        self.WindValues = np.zeros(shape=(params.NSamples, 3))
        
    def AddGust(self,Gust,GustTime):
        #cette fonction permet d'ajouter une gust sur la composante longitudinale
        # du signal de vent
        #self.WindValues[0,:]=self.WindValues[0,:]+Gust.GetGustSignal(self.params,GustTime)
        # Affichage (si decommente)
        #time=[0.0]*self.SimulationParameters.NSamples # def du vecteur de temps pour affichage
        #for i in range(len(time)):
        #    time[i]=float(i)*self.SimulationParameters.SampleTime
        #fig=plt.figure()
        #ax = fig.add_subplot(111)
        #ax.plot(time,self.WindValues[0,:])
        #plt.show()
        pass

     
    def compute(self, fft_seed=None, lidar_spectrum=None):
        Ts=self.params.SampleTime #Pertiode d'echantillonage
        N=self.params.NSamples

        # Création d'un spectre si aucun n'est donné.
        if lidar_spectrum is None:
            lidar_spectrum = LiDARSpectrum(self.params)
            _, spectrum = lidar_spectrum.compute(N, Ts*N)

        #initialisation des seeds si aucune n'est donnée en paramètre.
        if fft_seed is None:
            fft_seed = LiDARWind.get_initial_fftseed(N)
            
        # Multiplication du spectre de la seed, par le spectre (discret) du vent
        # Le spectre est defini comme le spectre original auquel s'ajoute son symetrique
        # pour pouvoir obtenir un spectre discret
        for i in range(self.wind_mean.shape[0]):
            wind_spectrum = np.multiply(
                fft_seed[:, i],
                spectrum[:, i]+spectrum[:, i][::-1]
                ) 
            
            self.WindValues[:, i] = np.fft.ifft(wind_spectrum).real + self.wind_mean[i]
            

       