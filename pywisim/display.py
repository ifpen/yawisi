import matplotlib.pyplot as plt
import numpy as np

from pywisim.spectrum import LiDARSpectrum
from pywisim.wind_field import LiDARWindField
from pywisim.locations import Grid
from pywisim.wind import LiDARWind


def display_spectrum(spectrum: LiDARSpectrum ):
    """
    Cette fonction permet de faire l'affichage du spectre 
    """

    freq, array = spectrum.compute()

    fig = plt.figure()
    fig.suptitle('Spectre du vent', fontsize=20)
    plt.xlabel('Frequence (Hz)', fontsize=18)
    plt.ylabel('Spectre', fontsize=16)
    plt.plot(freq, array[:, 0],label='Wx')
    plt.plot(freq, array[:, 1],label='Wy')
    plt.plot(freq, array[:, 2],label='Wz')
    plt.legend()
    plt.show()


def display_coherence_function(freq, coherence_function):
    plt.plot(freq, coherence_function)
    plt.ylabel('Coherence')
    plt.xlabel("Freq")
    plt.show()

  
def display_wind(wind: LiDARWind):
        #Cette fonction permet d'afficher le signal de vent contenu dans l'objet.
        # Si le vent n'a pas ete initialise, on visualise 0 pour tous les echantillons
       
        time = np.arange(start=0, stop=wind.params.total_time, step=wind.params.SampleTime)
        #affichage des trois composantes
        fig=plt.figure()
        ax = fig.add_subplot(111)
        #print self.WindValues
        #print self.WindValues[0,:]
        fig.suptitle('Vent genere', fontsize=20)
        ax.plot(time,wind.WindValues[:, 0],label='wx')
        ax.plot(time,wind.WindValues[:, 1],label='wy')
        ax.plot(time,wind.WindValues[:, 2],label='wz')
        plt.ylabel('Vent en m/s')
        plt.xlabel("Temps")
        plt.legend()
        plt.show()


def display_points(grid: Grid):
    #Affichage des points du champ simple, sans description des parametres
    fig=plt.figure()
    fig.suptitle('Points du Champ de vent', fontsize=20)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Position transversale')
    ax.set_ylabel('Position Verticale')
    ax.plot(grid.x_array(), grid.y_array(),'x')
    ax.grid()
    plt.show()
    

def display_field(wind_field: LiDARWindField):
    # Fonction pour Affichage des parametres du champ et
    # des points de generation du vent
    print('_______________Wind Field Display___________________________________')
    print('Simulation Parameters:')
    print('Samples Numbers: %s' % wind_field.params.NSamples)
    print('SampleTime: %s' % wind_field.params.SampleTime)
    #Affichage des points dans le fenetre de commande,
    #display des points dans une figure
    print('WindField Points:')
    for i in range(len(wind_field.Points)):
        pt = wind_field.Points.points[i]
        print('Point %s : Y=%s, Z=%s' % (i,pt[0],pt[1]))
        
    fig=plt.figure()
    fig.suptitle('Vent genere', fontsize=20)
    # Affichage des signaux de vent si ils ont ete initialises
    if not wind_field.is_initialized:
        print('Warning: Wind Values have not been initialized')
    else:
        print('Wind Values have been initialized')
        Time=[]
        ii=0
        ax2 = fig.add_subplot(111)
        while ii<wind_field.params.NSamples:
            Time.append(float(ii)*wind_field.params.SampleTime)
            ii+=1
        i=0
        while i<len(wind_field.Points):
            pt = wind_field.Points.points[i]
            ax2.plot(Time,wind_field.Wind[i].WindValues[:, 0],label='w_x , point[%s,%s]' % (pt[0],pt[1]))
            ax2.plot(Time,wind_field.Wind[i].WindValues[:, 0],'-',label='w_y , point[%s,%s]' % (pt[0],pt[1]) )
            ax2.plot(Time,wind_field.Wind[i].WindValues[:, 1],'.',label='w_z , point[%s,%s]' % (pt[0],pt[1]) )
            i+=1    
        ax2.set_ylabel('Vent(m/s)')
        ax2.set_xlabel("Temps")
        plt.legend()
        plt.grid ()    
        plt.show()

        # Fin de la fonction, cloture de l'affichage dans le script
    print('________________End OF DISPLAY____________________________________')