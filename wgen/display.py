import matplotlib.pyplot as plt
import numpy as np

from wgen.spectrum import LiDARSpectrum
from wgen.wind_field import LiDARWindField
from wgen.locations import Grid


def display_spectrum(spectrum: LiDARSpectrum, Npts=1000, FMax=4.0 ):
    """
    Cette fonction permet de faire l'affichage du spectre defini sur Npts Points avec une
    une frequence max de FMax Hz
    """

    freq = np.arange(0, FMax, FMax/Npts)
    array = np.zeros(shape=(3, Npts))
    array[0, :] = spectrum.kernel(spectrum.SigmaX, freq)
    array[1, :] = spectrum.kernel(spectrum.SigmaY, freq)
    array[2, :] = spectrum.kernel(spectrum.SigmaZ, freq)

    fig = plt.figure()
    fig.suptitle('Spectre du vent', fontsize=20)
    plt.xlabel('Frequence (Hz)', fontsize=18)
    plt.ylabel('Spectre', fontsize=16)
    plt.plot(freq, array[0,:],label='Wx')
    plt.plot(freq, array[1,:],label='Wy')
    plt.plot(freq, array[2,:],label='Wz')
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
    print('Samples Numbers: %s' % wind_field.SimulationParameters.NSamples)
    print('SampleTime: %s' % wind_field.SimulationParameters.SampleTime)
    #Affichage des points dans le fenetre de commande,
    #display des points dans une figure
    print('WindField Points:')
    i=0
    while i<len(wind_field.Points):
        print('Point %s : Y=%s, Z=%s' % (i,wind_field.Points[i][0],wind_field.Points[i][1]))
        i+=1
    fig=plt.figure()
    fig.suptitle('Vent genere', fontsize=20)
    # Affichage des signaux de vent si ils ont ete initialises
    if wind_field.WindValuesInitialized==0:
        print('Warning: Wind Values have not been initialized')
    else:
        print('Wind Values have been initialized')
        Time=[]
        ii=0
        ax2 = fig.add_subplot(111)
        while ii<wind_field.SimulationParameters.NSamples:
            Time.append(float(ii)*wind_field.SimulationParameters.SampleTime)
            ii+=1
        i=0
        while i<len(wind_field.Points):
            ax2.plot(Time,wind_field.Wind[i].WindValues[0,:],label='w_x , point[%s,%s]' % (wind_field.Points[i][0],wind_field.Points[i][1]))
            ax2.plot(Time,wind_field.Wind[i].WindValues[1,:],'-',label='w_y , point[%s,%s]' % (wind_field.Points[i][0],wind_field.Points[i][1]) )
            ax2.plot(Time,wind_field.Wind[i].WindValues[2,:],'.',label='w_z , point[%s,%s]' % (wind_field.Points[i][0],wind_field.Points[i][1]) )
            i+=1    
        ax2.set_ylabel('Vent(m/s)')
        ax2.set_xlabel("Temps")
        plt.legend()
        plt.grid ()    
        plt.show()

        # Fin de la fonction, cloture de l'affichage dans le script
    print('________________End OF DISPLAY____________________________________')