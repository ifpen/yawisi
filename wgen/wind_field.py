from wgen.parameters import LiDARSimulationParameters


class LiDARWindField:
    # CLASSE MAJEURE POUR LA SIMULATION
    #cette classe permet de definir un champ de vent contenant un certain nombre de points,
    # et permet de generer le vecteur de vent
    def __init__(self,params: LiDARSimulationParameters):
        self.Points=[] #Points du champ de vent
        self.Wind=[]   #Objets vent contenus dans le champ
        self.Spectrum=LiDARSpectrum() #Spectre du signal de vent
        self.Grid=[] # Initialisation des points de la grille
        # Detection du spectre pour le champ
        if not params.FileName:
            print('Spectrum for the Wind Field is standard') #Champ standard si pas de detection
        else:
            print(['Spectrum Inititialzed from',params.FileName])
        self.Spectrum.init_spectrum_from_text(params.FileName) #Initialisation a partir du fichier
        self.SimulationParameters=params #Def des parametres de simulation pour le Wind Field
        self.WindValuesInitialized=0 # Flag pour l'initialisation des valeurs de vent
    
    
    def AddPoints(self,Y,Z):
        # Cette fonction permet d'ajouter des points au champ. Cette fonction est majeure, puisqu'elle permet 
        # de definir les points pour lequels le vent va etre genere.
        #TEST de l'existence du point qu'on souhaite ajouter.
        ii=0
        flag=0
        while ii<len(self.Points):
            if ((float(Y)-self.Points[ii][0])**2+(float(Z)-self.Points[ii][1])**2)**(1./2)<=0.01:
                #print('Point (%s,%s) existing @ indice %s' %(Y,Z,ii))
                flag=1
                break
            else:
                ii+=1
        # SI ce point n'est pas detecte, alors on ajout le point en fin de liste
        if not flag==1:
            self.Points.append([float(Y),float(Z)])
            self.Wind.append(LiDARWind(self.SimulationParameters))
            self.WindValuesInitialized=0
        return ii #renvoi de l'indice, pour la connaissance dans le champ de vent du point correspondant
        #print(self.Points)
        
   
    def GetDistanceMatrix(self):
        #Cette fonction permet de recuperer la matrice des distances.
        #le termes [i,j] contient la distance du point d'indice i au point d'indice j
        i=0
        Matrix=array([[float(0)]*len(self.Points)]*len(self.Points))#initialisation
        #print('Distance Matrix:')
        while i<len(self.Points):#remplissage de la matrice
            j=0
            while j<len(self.Points):
                Matrix[i][j]=((float(self.Points[i][0])-float(self.Points[j][0]))**2+(float(self.Points[i][1])-float(self.Points[j][1]))**2)**(1./2)    
                j+=1
            #print(Matrix[i,:])
            i+=1
        return Matrix
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