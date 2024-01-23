

class LiDARSimulationParameters:

    def __init__(self, filename):
        #Initialisation
        self.Filename = filename #Fichier d'initialisation
        self.__parse()

    def __parse(self):
        #Initialisation a partir d'un fichier texte
        with open(self.Filename,"r") as file: 
            data=file.readlines() #lecture du fichier
            print('Simulation Initialized')
            self.n_samples=[int(s) for s in data[4].split() if s.isdigit()][0] #Nombre d'echantillons
            self.sample_time=float(data[5].split('\t')[0]) #Periode d'echantillonnnage
            self.wind_mean=[int(s) for s in data[8].split() if s.isdigit()][0] #Lecture de la vitesse moyenne
            self.Lv=[int(s) for s in data[9].split() if s.isdigit()][0] #Lecture de la longueur d'onde
            self.sigma_x=float(data[10].split('\t')[0]) #Lecture de la variance longitudinale
            self.sigma_y=float(data[11].split('\t')[0]) #Lecture de la variance Transversale
            self.sigma_z=float(data[12].split('\t')[0]) #Lecture de la variance longitudinale
            self.vertical_shear=float(data[16].split('\t')[0]) #Lecture du parametre de gradient
            self.reference_height=float(data[15].split('\t')[0]) #Lecture de la hauteur de reference du champ de vent
            self.grid_width=float(data[17].split('\t')[0])
            self.grid_height=float(data[18].split('\t')[0])
            self.grid_length= int(data[19].split('\t')[0])

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
