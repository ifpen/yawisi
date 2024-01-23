import configparser

class LiDARSimulationParameters:

    def __init__(self, filename):
        #Initialisation
        self.Filename = filename #Fichier d'initialisation
      
        self.n_samples = 1000
        self.sample_time = 0.1
        self.wind_mean = 10. #Lecture de la vitesse moyenne
        self.kind = 'karman'
        self.Lv = 130 #Lecture de la longueur d'onde
        self.sigma_x = 4. #Lecture de la variance longitudinale
        self.sigma_y = 2.8 #Lecture de la variance Transversale
        self.sigma_z = 1.2 #Lecture de la variance longitudinale
        self.vertical_shear= 0.3 #Lecture du parametre de gradient
        self.reference_height = 80 #Lecture de la hauteur de reference du champ de vent
        self.grid_width=100
        self.grid_height=100
        self.grid_length=11

        if filename is not None:
            self.__parse_ini(filename)

    def __parse_ini(self, filename):

        config = configparser.ConfigParser()
        config.read(filename)
        simulation = config['Simulation']
        self.n_samples = int(simulation.get('n_samples', self.n_samples))
        self.sample_time=float(simulation.get('sample_time', self.sample_time))

        spectrum = config['Spectrum']
        self.kind = spectrum.get('kind', self.kind)
        self.wind_mean= int(spectrum.get('wind_mean', self.wind_mean)) #Lecture de la vitesse moyenne
        self.Lv=int(spectrum.get('wave_length')) #Lecture de la longueur d'onde
        self.sigma_x=float(spectrum.get('sigma_x', self.sigma_x))
        self.sigma_y=float(spectrum.get('sigma_y', self.sigma_y))
        self.sigma_z=float(spectrum.get('sigma_z', self.sigma_z))

        field = config['Field']
        self.vertical_shear=float(field.get('vertical_shear', self.vertical_shear))
        self.reference_height=float(field.get('hub_height', self.reference_height))
        self.grid_width=float(field.get('grid_width', self.grid_width))
        self.grid_height=float(field.get('grid_heigth', self.grid_height))
        self.grid_length= int(field.get('grid_length', self.grid_length))

    @property
    def total_time(self):
        return self.n_samples*self.sample_time
    
    @property
    def freq_max(self):
        return 1. / self.sample_time

    def __str__(self) -> str:
        msg = f'Number of Samples Initialized @ {self.n_samples}\n'
        msg += f'Sample Time Initialized @ {self.sample_time}\n'
        msg = f'Wind Mean Speed Initialized @ {self.wind_mean}\n'
        msg +=f'Wave length Initialized @ {self.Lv}\n'
        msg +=f'Longitudinal Component Variance initialized @ {self.sigma_x}\n'
        msg +=f'Transversal Component Variance initialized @  {self.sigma_y}\n'
        msg +=f'Vertical Component Variance initialized @ {self.sigma_z}\n'
        msg +=f'PL Exp Law initialized  @ {self.vertical_shear}\n'
        msg +=f'Reference Height @ {self.reference_height}\n'
        msg +=f'Grid Width @ {self.grid_width}\n'
        msg +=f'Grid Height @ {self.grid_width}\n'
        msg +=f'Grid Length @ {self.grid_length}\n'
        return msg
