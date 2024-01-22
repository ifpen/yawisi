import numpy as np

from pywisim.parameters import LiDARSimulationParameters
from pywisim.kernels import Kaimal

class LiDARSpectrum:

    def __init__(self, params:LiDARSimulationParameters):
        self.params = params
        self.kernel = Kaimal(self)
        

    def compute(self, Npts=1000, FMax=4.0):
        freq = np.arange(0, FMax, FMax/Npts)
        array = np.zeros(shape=(Npts, 3))
        array[:, 0] = self.kernel(self.SigmaX, freq)
        array[:, 1] = self.kernel(self.SigmaY, freq)
        array[:, 2] = self.kernel(self.SigmaZ, freq)

        return freq, array 

    @property
    def SigmaX(self):
        return self.params.SigmaX
    
    @property
    def SigmaY(self):
        return self.params.SigmaY

    @property
    def SigmaZ(self):
        return self.params.SigmaZ
    
    @property
    def SigmaY(self):
        return self.params.SigmaY

    @property
    def SigmaZ(self):
        return self.params.SigmaZ
    
    @property
    def Lv(self):
        return self.params.Lv

    @property
    def WindMean(self):
        return self.params.WindMean
    
    @property
    def ReferenceHeight(self):
        return self.params.ReferenceHeight
    
    @property
    def VerticalShearParameter(self):
        return self.params.VerticalShearParameter
    
    def GenerateSignalFromSeedFFT(self, SeedFFT, str, Points, SpectrumScaling=6):
        """
        FONCTION MAJEURE POUR LA SIMULATION DE CHAMP DE VENT
        Cette fonction permet de generer le vecteur de vent a partir d'une seed
        deja passe dans le domaine frequentiel.
        cette fonction est utilisee pour la generation des signaux de vent une fois 
        que la coherence a ete initialisee

        Parameters
        ----------
        SpectrumScaling : float, optional
            Parametre de mise a l'echelle pour le spectre. Arbitraire (default is 6)

        """
        Ts=self.params.SampleTime #periode d'echantillonage
        N=self.params.NSamples #nombre de points
        freq=[] #initialisation du vecteur des frequences
         
        Spectrum=array([0.0]*N) #Spectre 
        #print(N)
        Time=[] #Vecteiur de temps
        i=0
        while i<N:
            #Boucle pour la definittion des grandeurs
            freq.append(float(i)/Ts/N) #vecteur frequences
            Time.append(float(i)*Ts) #vecteur temps
            #Initialisation des spectres selon la composante consideree
            if str=='X':
                Spectrum[i]=(SpectrumScaling*(self.SigmaX**2)*self.Lv/float(self.WindMean))/(1+2*3.14159*(float(i)/float(N)/Ts)*self.Lv/float(self.WindMean))**(float(5./3))
            elif str=='Y':
                Spectrum[i]=(SpectrumScaling*(self.SigmaY**2)*self.Lv/float(self.WindMean))/(1+2*3.14159*(float(i)/float(N)/Ts)*self.Lv/float(self.WindMean))**(float(5./3))
            elif str=='Z':
                Spectrum[i]=(SpectrumScaling*(self.SigmaZ**2)*self.Lv/float(self.WindMean))/(1+2*3.14159*(float(i)/float(N)/Ts)*self.Lv/float(self.WindMean))**(float(5./3))
            i+=1
        WindSpectrum=multiply(SeedFFT,Spectrum+Spectrum[::-1]) # multiplication de la seed par le spectre du vent
        if str=='X':
                WindMeanSpeed=self.WindMean*((self.ReferenceHeight+Points[1])/self.ReferenceHeight)**(self.VerticalShearParameter)
                WindValuesOUT=array(fft.ifft(WindSpectrum).real+WindMeanSpeed)
        elif str=='Y':
                WindValuesOUT=array(fft.ifft(WindSpectrum).real)
        elif str=='Z':
                WindValuesOUT=array(fft.ifft(WindSpectrum).real)
        #renvoi des valeurs de vent sous la forme d'un vecteur
        return WindValuesOUT



    def GenerateSignal(self):
        #cette fonction genere un signal aleatoire de vent pour les parametres de 
        #simulation demandes
        N = self.params.NSamples
        Ts = self.params.SampleTime
        freq=[]
        Spectrum=array([[0.0]*N]*3)
        seed=array([[0.0]*N]*3)
        Time=[]
        i=0
        while i<N:
            #Cette boucle definit les valeurs des vecteurs de temps et de frequences
            freq.append(float(i)/Ts/N)
            Time.append(float(i)*Ts)
            #Definition des seed pour la generation
            seed[0,i]=random.normal(loc=0.0, scale=1.0, size=None)
            seed[1,i]=random.normal(loc=0.0, scale=1.0, size=None)
            seed[2,i]=random.normal(loc=0.0, scale=1.0, size=None)
            #Definition des trois composantes du spectre
            Spectrum[0,i]=(0.475*(self.SigmaX**2)*self.Lv/self.WindMean)/(1+(2*3.14159*float(i)/float(N)/Ts*self.Lv/self.WindMean)**2)**(float(5/3))
            Spectrum[1,i]=(0.475*(self.SigmaY**2)*self.Lv/self.WindMean)/(1+2*3.14159*(float(i)/float(N)/Ts)*self.Lv/self.WindMean)**(float(5/3))
            Spectrum[2,i]=(0.475*(self.SigmaZ**2)*self.Lv/self.WindMean)/(1+2*3.14159*(float(i)/float(N)/Ts)*self.Lv/self.WindMean)**(float(5/3))
            i+=1
            # Multiplication du spectre de la seed, par le spectre (discret) du vent
            # Le spectre est defini comme le spectre original auquel s'ajoute son symetrique
            # pour pouvoir obtenir un spectre discret
        WindSpectrumX=multiply(fft.fft(seed[0,:]),Spectrum[0,:]+Spectrum[0,:][::-1])
        WindSpectrumY=multiply(fft.fft(seed[1,:]),Spectrum[1,:]+Spectrum[1,:][::-1])
        WindSpectrumZ=multiply(fft.fft(seed[2,:]),Spectrum[2,:]+Spectrum[2,:][::-1])
        
        #Affichage du vent genere
        plt.plot(Time,fft.ifft(WindSpectrumX).real+self.WindMean,label='wx')
        plt.plot(Time,fft.ifft(WindSpectrumY).real,label='wy')
        plt.plot(Time,fft.ifft(WindSpectrumZ).real,label='wz')
        plt.ylabel('Vent en m/s')
        plt.xlabel("Temps")
        plt.legend()
        plt.show()
        
   