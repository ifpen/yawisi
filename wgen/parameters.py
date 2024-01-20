

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
            self.NSamples=[int(s) for s in data[4].split() if s.isdigit()][0] #Nombre d'echantillons
            self.SampleTime=float(data[5].split('\t')[0]) #Periode d'echantillonnnage
            self.WindMean=[int(s) for s in data[8].split() if s.isdigit()][0] #Lecture de la vitesse moyenne
            self.Lv=[int(s) for s in data[9].split() if s.isdigit()][0] #Lecture de la longueur d'onde
            self.SigmaX=float(data[10].split('\t')[0]) #Lecture de la variance longitudinale
            self.SigmaY=float(data[11].split('\t')[0]) #Lecture de la variance Transversale
            self.SigmaZ=float(data[12].split('\t')[0]) #Lecture de la variance longitudinale
            self.VerticalShearParameter=float(data[16].split('\t')[0]) #Lecture du parametre de gradient
            self.ReferenceHeight=float(data[15].split('\t')[0]) #Lecture de la hauteur de reference du champ de vent
            self.GridWidth=float(data[17].split('\t')[0])
            self.GridHeight=float(data[18].split('\t')[0])
            self.GridLength= int(data[19].split('\t')[0])


    def __str__(self) -> str:
        msg = f'Number of Samples Initialized @ {self.NSamples}\n'
        msg += f'Sample Time Initialized @ {self.SampleTime}\n'
        msg = f'Wind Mean Speed Initialized @ {self.WindMean}\n'
        msg +=f'Wave length Initialized @ {self.Lv}\n'
        msg +=f'Longitudinal Component Variance initialized @ {self.SigmaX}\n'
        msg +=f'Vertical Component Variance initialized @ {self.SigmaZ}\n'
        msg +=f'PL Exp Law initialized  @ {self.VerticalShearParameter}\n'
        msg +=f'Reference Height @ {self.ReferenceHeight}\n'
        msg +=f'Grid Width @ {self.GridWidth}\n'
        msg +=f'Grid Height @ {self.GridWidth}\n'
        msg +=f'Grid Length @ {self.GridLength}\n'
        return msg
