import matplotlib.pyplot as plt #Plotting functions
from pylab import * #Mathematical functions
from numpy import random,array,fft,multiply,linalg,dot,pi #Mathematical functions
from scipy.io import savemat #Interfacing with Matlab
from scipy import interpolate #Interpolation function for the LIDAR simulation
from math import exp,atan #exp(0)=1 :-)
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
from matplotlib import animation # For Animation
import os # Basic function to control the OS using a Python script.


class LiDARSimulationParameters:
	def __init__(self,N,Ts):
		#Initialisation
		self.NSamples=N #Nombre d'echantillons
		self.SampleTime=Ts #Periode d'echantillonnnage
		self.FileName=[] #Fichier d'initialisation
	def InitFromText(self,str):
		#Initialisation a partir d'un fichier texte
		self.FileName=str #Set le fichier texte
		file=open(str,"r") #ouverture du fichier
		data=file.readlines() #lecture du fichier
		print('Simulation Initialized')
		self.NSamples=[int(s) for s in data[4].split() if s.isdigit()][0]
		print('Number of Samples Initialized @ %s ' % self.NSamples)
		self.SampleTime=float(data[5].split('\t')[0])
		print('Sample Time Initialized @ %s ' % self.SampleTime)
		file.close() #fermeture du fichier
		#Fin de l'initialisation des parametres de simulation
		
class LiDARSpectrum:
	def __init__(self):
		self.WindMean=[] #Mean Value of the Wind @ hub Height
		self.Lv=[] #Wave length
		self.SigmaX=[] #Variance of the X seed
		self.SigmaY=[] #Variance of the Y seed
		self.SigmaZ=[] #Variance of the Z seed
		self.VerticalShearParameter=[] #Shear parameter for the exponentia law
		self.ReferenceHeight=[]	#Reference height for the hub. Used for shear def.
	def InitSpectrumFromText(self,str):
		#Initialize the Spectrum parameters
		file=open(str,"r") #Ouverture fichier
		data=file.readlines() #Lecture 
		self.WindMean=[int(s) for s in data[8].split() if s.isdigit()][0] #Lecture de la vitesse moyenne
		print('Wind Mean Speed Initialized @ %s ' % self.WindMean)
		self.Lv=[int(s) for s in data[9].split() if s.isdigit()][0] #Lecture de la longueur d'onde
		print('Wave length Initialized @ %s ' % self.Lv)
		self.SigmaX=float(data[10].split('\t')[0]) #Lecture de la variance longitudinale
		print('Longitudinal Component Variance initialized @ %s' % self.SigmaX)
		self.SigmaY=float(data[11].split('\t')[0]) #Lecture de la variance Transversale
		print('Transversal Component Variance initialized @ %s' % self.SigmaY)
		self.SigmaZ=float(data[12].split('\t')[0]) #Lecture de la variance longitudinale
		print('Vertical Component Variance initialized @ %s' % self.SigmaZ)
		self.VerticalShearParameter=float(data[16].split('\t')[0]) #Lecture du parametre de gradient
		print('PL Exp Law initialized  @ %s' % self.VerticalShearParameter)
		self.ReferenceHeight=float(data[15].split('\t')[0]) #Lecture de la hauteur de reference du champ de vent
		print('Reference Height @ %s' % self.ReferenceHeight)
	def DisplaySpectrum(self):
		#Cette fonction permet de faire l'affichage du spectre defini sur 1000 Points avec une
		#Avec une frequence max de 10 Hz
		i=0 
		Npts=1000 
		FMax=4.0
		freq=[] #Initialisation du vecteur des frequences
		Spectrum=array([[0.0]*Npts]*3) #Initialisation du Spectre (3D)
		while i<Npts:
			#  Boucle definissant les valeurs du spectre pour les trois composantes
			freq.append(float(i)/float(Npts)*FMax)
			Spectrum[0,i]=(0.475*(self.SigmaX**2)*self.Lv/self.WindMean)/(1+2*3.14159*(float(i)/float(Npts)*FMax)*self.Lv/self.WindMean)**(float(5/3))
			Spectrum[1,i]=(0.475*(self.SigmaY**2)*self.Lv/self.WindMean)/(1+2*3.14159*(float(i)/float(Npts)*FMax)*self.Lv/self.WindMean)**(float(5/3))
			Spectrum[2,i]=(0.475*(self.SigmaZ**2)*self.Lv/self.WindMean)/(1+2*3.14159*(float(i)/float(Npts)*FMax)*self.Lv/self.WindMean)**(float(5/3))
			#Affichage
			i+=1
		fig = plt.figure()
		fig.suptitle('Spectre du vent', fontsize=20)
		plt.xlabel('Frequence (Hz)', fontsize=18)
		plt.ylabel('Spectre', fontsize=16)
		plt.plot(freq,Spectrum[0,:],label='Wx')
		plt.plot(freq,Spectrum[1,:],label='Wy')
		plt.plot(freq,Spectrum[2,:],label='Wz')
		plt.legend()
		plt.show()
		
	def GenerateSignal(self,SimulationParameters):
		#cette fonction genere un signal aleatoire de vent pour les parametres de 
		#simulation demandes
		N=SimulationParameters.NSamples
		Ts=SimulationParameters.SampleTime
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
		
	def GenerateHHShear(self,Wind,str):
		# Cette fonction permet de generer un fichier .HH qui peut ensuite etre utilise acvec le simualteur
		# FAST developpe au NREL. Le fichier est genere dans le dossier dexecution du script
		# sous le nom WindGenerated.HH
		# ATTENTION CETTE FONCTION EST INCOMPLETE
		Ts=Wind.SimulationParameters.SampleTime # definition de la periode d'echantilonnage
		file=open(str,"w") #ouvertur du fichier
		file.write('! Wind file for Trivial turbine.\n')
		file.write('! Time	Wind	Wind	Vert.	Horiz.	Vert.	LinV	Gust\n')
		file.write('!	Speed	Dir	Speed	Shear	Shear	Shear	Speed\n')
		#Generation du vecteur de vent a partir de la seed 
		OUT=self.GenerateSignalFromWind(Wind)
		#print(OUT)
		i=0
		while i<len(Wind.SeedX):
			#Boucle d'ecriture dans le fichier .HH
			Time=float(i)*Ts
			file.write("%.3f  " % Time)
			file.write("%.3f  " % (OUT[0,i]**2+OUT[1,i]**2)**(0.5))
			Angle=180.*(pi)**(-1)*atan(OUT[1,i]/OUT[0,i])
			print("Angle=%.3f" % Angle)
			file.write("%.3f  " % -Angle)
			file.write("%.3f  " % OUT[2,i])
			file.write("0  %.3f  " % self.VerticalShearParameter)
			file.write('0 0 0 \n')
			i+=1
		file.close() #Fermeture du fichier
	def GenerateHH(self,Wind,str):
		# Cette fonction permet de generer un fichier .HH qui peut ensuite etre utilise acvec le simualteur
		# FAST developpe au NREL. Le fichier est genere dans le dossier dexecution du script
		# sous le nom WindGenerated.HH
		# ATTENTION CETTE FONCTION EST INCOMPLETE
		Ts=Wind.SimulationParameters.SampleTime # definition de la periode d'echantilonnage
		file=open(str,"w") #ouvertur du fichier
		file.write('! Wind file for Trivial turbine.\n')
		file.write('! Time	Wind	Wind	Vert.	Horiz.	Vert.	LinV	Gust\n')
		file.write('!	Speed	Dir	Speed	Shear	Shear	Shear	Speed\n')
		#Generation du vecteur de vent a partir de la seed 
		OUT=self.GenerateSignalFromWind(Wind)
		#print(OUT)
		i=0
		while i<len(Wind.SeedX):
			#Boucle d'ecriture dans le fichier .HH
			Time=float(i)*Ts
			file.write("%.3f  " % Time)
			file.write("%.3f  " % (OUT[0,i]**2+OUT[1,i]**2)**(0.5))
			Angle=180.*(pi)**(-1)*atan(OUT[1,i]/OUT[0,i])
			#print("Angle=%.3f" % Angle)
			file.write("%.3f  " % -Angle)
			file.write("%.3f  " % OUT[2,i])
			file.write("0  %.3f  " % 0)
			file.write('0 0 0 \n')
			i+=1
		file.close() #Fermeture du fichier

	def GenerateSignalFromWind(self,Wind):

		#Cette Fonction permet de recuperer le vent genere a partir d'une seed de vent.
		Ts=Wind.SimulationParameters.SampleTime #Pertiode d'echantillonage
		N=Wind.SimulationParameters.NSamples
		freq=[]
		Spectrum=array([[0.0]*N]*3)
		print(N)
		Time=[]
		i=0
		while i<N:
			#Definition des vecteur de frequence et de temps
			freq.append(float(i)/Ts/N)
			Time.append(float(i)*Ts)
			#definition des spectres
			Spectrum[0,i]=(2*(self.SigmaX**2)*self.Lv/self.WindMean)/(1+2*3.14159*(float(i)/float(N)/Ts)*self.Lv/self.WindMean)**(float(5/3))
			Spectrum[1,i]=(2*(self.SigmaY**2)*self.Lv/self.WindMean)/(1+2*3.14159*(float(i)/float(N)/Ts)*self.Lv/self.WindMean)**(float(5/3))
			Spectrum[2,i]=(2*(self.SigmaZ**2)*self.Lv/self.WindMean)/(1+2*3.14159*(float(i)/float(N)/Ts)*self.Lv/self.WindMean)**(float(5/3))
			i+=1
			#multiplication des spectres
		WindSpectrumX=multiply(fft.fft(Wind.SeedX),Spectrum[0,:]+Spectrum[0,:][::-1])
		WindSpectrumY=multiply(fft.fft(Wind.SeedY),Spectrum[1,:]+Spectrum[1,:][::-1])
		WindSpectrumZ=multiply(fft.fft(Wind.SeedZ),Spectrum[2,:]+Spectrum[2,:][::-1])
		plt.plot(Spectrum[0,:])
		plt.show()
		#Affichage
		plt.plot(Time,fft.ifft(WindSpectrumX).real+self.WindMean,label='wx')
		plt.plot(Time,fft.ifft(WindSpectrumY).real,label='wy')
		plt.plot(Time,fft.ifft(WindSpectrumZ).real,label='wz')
		plt.ylabel('Vent en m/s')
		plt.xlabel("Temps")
		plt.legend()
		plt.show()
		#renvoie les valeurs de vent
		return array([fft.ifft(WindSpectrumX).real+self.WindMean,fft.ifft(WindSpectrumY).real,fft.ifft(WindSpectrumZ).real])
		
	def GenerateSignalFromSeedFFT(self,SeedFFT,SimulationParameters,str,Points):
		# FONCTION MAJEURE POUR LA SIMULATION DE CHAMP DE VENT
		#Cette fonction permet de generer le vecteur de vent a partir d'une seed
		# deja passe dans le domaine frequentiel.
		# cette fonction est utilisee pour la generation des signaux de vent une fois 
		# que la coherence a ete initialisee
		Ts=SimulationParameters.SampleTime #periode d'echantillonage
		N=SimulationParameters.NSamples #nombre de points
		freq=[] #initialisation du vecteur des frequences
		SpectrumScaling=6 # Parametre de mise a l'echelle pour le spectre. Arbitraire
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

class LiDARWind:
	# Cette classe permet de definir un objet Vent, contenant une seed (pour chaque composante)
	# Elle permet egalement de contenir les valeurs du vent, qui peuvent etre initialisee 
	# a partir de la classe spectre, ou par la fonction du champ de vent.
	def __init__(self,SimulationParameters):
	#initialisation des seeds a  0 et du vent a 0
		self.SeedX=array([0.0]*SimulationParameters.NSamples)
		self.SeedY=array([0.0]*SimulationParameters.NSamples)
		self.SeedZ=array([0.0]*SimulationParameters.NSamples)
		self.WindValues=array([[0.0]*SimulationParameters.NSamples]*3)
		self.SimulationParameters=SimulationParameters
		#initialisation des seeds
		i=0
		while i<SimulationParameters.NSamples:
			#modelisation des seeds comme une variable gaussienne.
			self.SeedX[i]=(random.normal(loc=0.0, scale=1.0, size=None))
			self.SeedY[i]=(random.normal(loc=0.0, scale=1.0, size=None))
			self.SeedZ[i]=(random.normal(loc=0.0, scale=1.0, size=None))
			i+=1
		#Mise a zero de la moyenne de la seed
		self.SeedX=self.SeedX-(sum(self.SeedX)/SimulationParameters.NSamples)
		self.SeedY=self.SeedY-(sum(self.SeedX)/SimulationParameters.NSamples)
		self.SeedZ=self.SeedZ-(sum(self.SeedX)/SimulationParameters.NSamples)
		#fin de l'initialisation.
	def DisplayWind(self):
		#Cette fonction permet d'afficher le signal de vent contenu dans l'objet.
		# Si le vent n'a pas ete initialise, on visualise 0 pour tous les echantillons
		time=[0.0]*self.SimulationParameters.NSamples #Definition du vecteur de temps
		for i in range(len(time)):
			time[i]=float(i)*self.SimulationParameters.SampleTime
		#affichage des trois composantes
		fig=plt.figure()
		ax = fig.add_subplot(111)
		#print self.WindValues
		#print self.WindValues[0,:]
		fig.suptitle('Vent genere', fontsize=20)
		ax.plot(time,self.WindValues[0,:],label='wx')
		ax.plot(time,self.WindValues[1,:],label='wy')
		ax.plot(time,self.WindValues[2,:],label='wz')
		plt.ylabel('Vent en m/s')
		plt.xlabel("Temps")
		plt.legend()
		plt.show()	
	def AddGust(self,Gust,GustTime):
		#cette fonction permet d'ajouter une gust sur la composante longitudinale
		# du signal de vent
		self.WindValues[0,:]=self.WindValues[0,:]+Gust.GetGustSignal(self.SimulationParameters,GustTime)
		# Affichage (si decommente)
		#time=[0.0]*self.SimulationParameters.NSamples # def du vecteur de temps pour affichage
		#for i in range(len(time)):
		#	time[i]=float(i)*self.SimulationParameters.SampleTime
		#fig=plt.figure()
		#ax = fig.add_subplot(111)
		#ax.plot(time,self.WindValues[0,:])
		#plt.show()
	def SetWind(self,Spectrum):
		self.WindValues=Spectrum.GenerateSignalFromWind(self)
	def GenerateMat(self,str):
		DataStructure=dict()
		WindX=self.WindValues[0,:]
		WindY=self.WindValues[1,:]
		WindZ=self.WindValues[2,:]
		DataStructure["WindX"]=WindX
		DataStructure["WindY"]=WindY
		DataStructure["WindZ"]=WindZ
		savemat(str, DataStructure)
		
class LiDARWindGrid:
	# Cette classe permet de definir une grid de points pour executer le vent.
	def __init__(self,Width,Height,Npts):
		self.Points=[] # Points des la grille de vent
		self.Indices=array([[-1]*Npts]*Npts) #indices correspondant (utilise pour le champ de vent dans la simulation Lidar)
		WidthPoints=[] #Vecteur des abcisses
		HeightPoints=[] #Vecteur des ordonnees
		ii=0
		while ii<Npts:
			#Generation des points sur la largeur et la hauteur
			WidthPoints.append([ii/float(Npts-1)*Width-Width/2])
			HeightPoints.append([ii/float(Npts-1)*Height-Height/2])
			ii+=1
		#Creation d'une grille des points
		Y,Z=np.meshgrid(WidthPoints,HeightPoints)
		#print(Y)
		#def des points sous la forme d'une grille
		self.Points=[Y,Z]
		#print self.Points[1][0,0]
			
class LiDARWindField:
	# CLASSE MAJEURE POUR LA SIMULATION
	#cette classe permet de definir un champ de vent contenant un certain nombre de points,
	# et permet de generer le vecteur de vent
	def __init__(self,SimulationParameters):
		self.Points=[] #Points du champ de vent
		self.Wind=[]   #Objets vent contenus dans le champ
		self.Spectrum=LiDARSpectrum() #Spectre du signal de vent
		self.Grid=[] # Initialisation des points de la grille
		# Detection du spectre pour le champ
		if not SimulationParameters.FileName:
			print('Spectrum for the Wind Field is standard') #Champ standard si pas de detection
		else:
			print(['Spectrum Inititialzed from',SimulationParameters.FileName])
		self.Spectrum.InitSpectrumFromText(SimulationParameters.FileName) #Initialisation a partir du fichier
		self.SimulationParameters=SimulationParameters #Def des parametres de simulation pour le Wind Field
		self.WindValuesInitialized=0 # Flag pour l'initialisation des valeurs de vent
	def DisplayField(self):
		# Fonction pour Affichage des parametres du champ et
		# des points de generation du vent
		print('_______________Wind Field Display___________________________________')
		print('Simulation Parameters:')
		print('Samples Numbers: %s' % self.SimulationParameters.NSamples)
		print('SampleTime: %s' % self.SimulationParameters.SampleTime)
		#Affichage des points dans le fenetre de commande,
		#display des points dans une figure
		print('WindField Points:')
		i=0
		while i<len(self.Points):
			print('Point %s : Y=%s, Z=%s' % (i,self.Points[i][0],self.Points[i][1]))
			i+=1
		fig=plt.figure()
		fig.suptitle('Vent genere', fontsize=20)
		# Affichage des signaux de vent si ils ont ete initialises
		if self.WindValuesInitialized==0:
			print('Warning: Wind Values have not been initialized')
		else:
			print('Wind Values have been initialized')
			Time=[]
			ii=0
			ax2 = fig.add_subplot(111)
			while ii<self.SimulationParameters.NSamples:
				Time.append(float(ii)*self.SimulationParameters.SampleTime)
				ii+=1
			i=0
			while i<len(self.Points):
				ax2.plot(Time,self.Wind[i].WindValues[0,:],label='w_x , point[%s,%s]' % (self.Points[i][0],self.Points[i][1]))
				ax2.plot(Time,self.Wind[i].WindValues[1,:],'-',label='w_y , point[%s,%s]' % (self.Points[i][0],self.Points[i][1]) )
				ax2.plot(Time,self.Wind[i].WindValues[2,:],'.',label='w_z , point[%s,%s]' % (self.Points[i][0],self.Points[i][1]) )
				i+=1	
			ax2.set_ylabel('Vent(m/s)')
			ax2.set_xlabel("Temps")
			plt.legend()
			plt.grid ()	
			plt.show()

			# Fin de la fonction, cloture de l'affichage dans le script
		print('________________End OF DISPLAY____________________________________')
	
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
		
	def DisplayPoints(self):
		#Affichage des points du champ simple, sans description des parametres
		fig=plt.figure()
		fig.suptitle('Points du Champ de vent', fontsize=20)
		ax = fig.add_subplot(111)
		ax.set_xlabel('Position transversale')
		ax.set_ylabel('Position Verticale')
		ax.plot(array(self.Points)[:,0],array(self.Points)[:,1],'x')
		ax.grid()
		plt.show()
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
	def SetWindValues(self):
		N=self.SimulationParameters.NSamples
		Ts=self.SimulationParameters.SampleTime
		#Definition de la fonction de coherence et de la frequence
		i=0
		freq=[]
		CoherenceFonction=[]
		Time=[]
		while i<N:
			Time.append(float(i)*Ts)
			freq.append(float(i)/Ts/N)
			if i<N/2:
				CoherenceFonction.append(1*((2*3.14159*float(i)/Ts/N/100)**2+0.003**2)**(1./2))
			else:
				CoherenceFonction.append(1*((2*3.14159*float(N-i)/Ts/N/100)**2+0.003**2)**(1./2))
			i+=1
		#plt.plot(freq,CoherenceFonction)
		#plt.ylabel('Coherence')
		#plt.xlabel("Freq")
		#plt.show()
		#Definition des transformation de Fourier des seeds du vent
		i=0
		SeedFftX=[]
		SeedFftY=[]
		SeedFftZ=[]
		while i<len(self.Wind):
			SeedFftX.append(fft.fft(self.Wind[i].SeedX))
			SeedFftY.append(fft.fft(self.Wind[i].SeedY))
			SeedFftZ.append(fft.fft(self.Wind[i].SeedZ))
			i+=1
		SeedFftX=array(SeedFftX)
		SeedFftY=array(SeedFftY)
		SeedFftZ=array(SeedFftZ)
		#print(SeedFft)
		#Multiplication par la matrice de coherence
		i=0
		while i<N:
			CoherenceMatrice=self.GetCoherenceMatrix(CoherenceFonction[i])
			L=linalg.cholesky(CoherenceMatrice)
			#print(L)
			SeedFftX[:,i]=dot(L,SeedFftX[:,i])
			SeedFftY[:,i]=dot(L,SeedFftY[:,i])
			SeedFftZ[:,i]=dot(L,SeedFftZ[:,i])
			print('Coherent Seed initialized @ %s %%' % (i*100/float(N)))
			i+=1
		i=0
		while i<len(self.Points):
			self.Wind[i].WindValues[0,:]=self.Spectrum.GenerateSignalFromSeedFFT(SeedFftX[i,:],self.SimulationParameters,'X',self.Points[i])
			self.Wind[i].WindValues[1,:]=self.Spectrum.GenerateSignalFromSeedFFT(SeedFftY[i,:],self.SimulationParameters,'Y',self.Points[i])
			self.Wind[i].WindValues[2,:]=self.Spectrum.GenerateSignalFromSeedFFT(SeedFftZ[i,:],self.SimulationParameters,'Z',self.Points[i])
			i+=1	
		self.WindValuesInitialized=1
		
	def SetGrid(self,Width,Height,Npts):
		self.Grid=LiDARWindGrid(Width,Height,Npts)
		ii=0
		while ii<Npts:
			jj=0
			while jj<Npts:
				#print(self.Grid.Points[0][ii,jj])
				#print(self.Grid.Points[1][ii,jj])
				self.Grid.Indices[ii,jj]=(self.AddPoints(self.Grid.Points[0][ii,jj],self.Grid.Points[1][ii,jj]))
				jj+=1
			ii+=1
		#print(self.Grid.Indices)
	def DisplayWindGrid(self,index):
		#Second display:
		X=self.Grid.Points[0]
		#print(X)
		Y=self.Grid.Points[1]
		Z=array([[0]*X.shape[0]]*X.shape[1])
		#print(Y)
		ii=0
		while ii<X.shape[0]:
			jj=0
			while jj<X.shape[1]:
				Z[ii,jj]=self.Wind[self.Grid.Indices[ii,jj]].WindValues[0,index]
				#print(self.Wind[self.Grid.Indices[ii,jj]].WindValues[0,index])
				jj+=1
			ii+=1
		fig=plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlabel('Vent(m/s)')
		ax.set_ylabel('Position transversale')
		ax.set_zlabel('Position verticale')
		ax.plot_wireframe(Z,X,Y)
		plt.show()
	def GetWindGrid(self,index):
		#Second display:
		X=self.Grid.Points[0]
		#print(X)
		Y=self.Grid.Points[1]
		Z=array([[0]*X.shape[0]]*X.shape[1])
		#print(Y)
		ii=0
		while ii<X.shape[0]:
			jj=0
			while jj<X.shape[1]:
				Z[ii,jj]=self.Wind[self.Grid.Indices[ii,jj]].WindValues[0,index]
				#print(self.Wind[self.Grid.Indices[ii,jj]].WindValues[0,index])
				jj+=1
			ii+=1
		return X,Y,Z	
	def AddGust(self,Gust,GustTime):
		for i in range(len(self.Wind)):
			self.Wind[i].AddGust(Gust,GustTime)
			
class LiDARGust:
	def __init__(self,Amplitude,Width,Time):
	#initialisation de la Gust
		self.Amplitude=Amplitude
		self.Width=Width
		self.Time=Time
	def DisplayGust(self):
		Nsamples=500
		t=[0.0]*Nsamples
		Gust=[0.0]*Nsamples
		for i in range(Nsamples):
			t[i]=float(i)*self.Width/Nsamples-self.Width/2
			Gust[i]=-0.37*self.Amplitude*sin(3*pi*float(i)/Nsamples)*(1-cos(2*pi*float(i)/Nsamples))
		fig=plt.figure()
		fig.suptitle('Mexican Hat Gust')
		ax = fig.add_subplot(111)
		ax.plot(t,Gust,label='Gust')
		ax.set_xlabel('t(s)')
		ax.set_ylabel('Vent(m/s)')
		plt.grid()
		plt.legend()
		plt.show()
	def DisplayWindSignal(self,SimulationParameters,GustTime):
		#Creation d'un vecteur de temps
		time=[0.0]*SimulationParameters.NSamples
		Wind=array([0.0]*SimulationParameters.NSamples)
		for i in range(len(time)):
			time[i]=float(i)*SimulationParameters.SampleTime
			if abs(time[i]-GustTime)<self.Width/2:
				Wind[i]=0.37*self.Amplitude*sin(3*pi*(time[i]-GustTime-self.Width/2)/self.Width)*(1-cos(2*pi*(time[i]-GustTime-self.Width/2)/self.Width))
		fig=plt.figure()
		ax = fig.add_subplot(111)
		fig.suptitle('Mexican Hat Gust')
		ax.plot(time,Wind,label='Gust')
		ax.set_xlabel('t(s)')
		ax.set_ylabel('Vent(m/s)')
		plt.grid()
		plt.legend()
		plt.show()		
	def GetGustSignal(self,SimulationParameters,GustTime):
		#Creation d'un vecteur de temps
		time=[0.0]*SimulationParameters.NSamples
		Wind=array([0.0]*SimulationParameters.NSamples)
		for i in range(len(time)):
			time[i]=float(i)*SimulationParameters.SampleTime
			if abs(time[i]-GustTime)<self.Width/2:
				Wind[i]=0.37*self.Amplitude*sin(3*pi*(time[i]-GustTime-self.Width/2)/self.Width)*(1-cos(2*pi*(time[i]-GustTime-self.Width/2)/self.Width))
		return Wind
		

class LiDARMeasure:
	def __init__(self,MeasureParameters):
		self.MeasureParameters=MeasureParameters
		self.Theta=MeasureParameters[1]
		self.DistanceParameter=MeasureParameters[2]
		self.Phi=MeasureParameters[0]
		self.Distance=array([0.0]*61)
		self.GridPoint=array([[0.0]*2]*61)
		self.Indices=array([0]*61)
		self.Ponderation=array([0.0]*61)
		sumPond=0.0
		for i in range(61):
			self.Distance[i]=float(i)-float(30)+float(MeasureParameters[2])
			self.GridPoint[i,0]=self.Distance[i]*sin(MeasureParameters[0]*pi/180)*cos(MeasureParameters[1]*pi/180)
			self.GridPoint[i,1]=self.Distance[i]*sin(MeasureParameters[0]*pi/180)*sin(MeasureParameters[1]*pi/180)
			self.Ponderation[i]=float(30-abs(i-30))
			sumPond+=float(30-abs(i-30))
		self.Ponderation=self.Ponderation/sumPond
		print ('Nouvelle Mesure: Phi=%s,Theta=%s,d=%s' % (MeasureParameters[0],MeasureParameters[1],MeasureParameters[2]))
	def GetPoints(self):
		return self.GridPoint
		
class LiDARSensorSimulation:
	def __init__(self,SimulationParameters):
		print('---------Initialisation du Champ de vent--------')
		self.WindField=LiDARWindField(SimulationParameters)
		print('---------Champ de vent initialise---------')
		file=open(SimulationParameters.FileName,"r")
		data=file.readlines()
		print('---------Initialisation des mesures LiDAR--------')
		self.Measure=[]
		i=22
		while i<60:
			if data[i].split('\t')[0]=='END':
				break
			else:
				Mesure=[0]*3
				Mesure[0]=float(data[i].split('\t')[0])
				Mesure[1]=float(data[i].split('\t')[1])
				Mesure[2]=float(data[i].split('\t')[2])
				self.Measure.append(LiDARMeasure(Mesure))
				#print(i,Mesure)
			i+=1
		for i in range(len(self.Measure)):
			Points=self.Measure[i].GetPoints()
			for j in range(len(Points)):
				self.Measure[i].Indices[j]=self.WindField.AddPoints(Points[j,0],Points[j,1])
		print('---------Mesures LiDAR initialisee--------')
		print('---------Initialisation de la grille de vent--------')
		print('GridWidth=%s' % float(data[17].split('\t')[0]))
		print ('GridHeight=%s' % float(data[18].split('\t')[0]))
		print ('GridLength=%s' % int(data[19].split('\t')[0]))
		self.WindField.SetGrid(float(data[17].split('\t')[0]),float(data[18].split('\t')[0]),int(data[19].split('\t')[0]))
		#self.WindField.DisplayPoints()
		print('---------Grid Initialized--------')
	def GenerateMat(self,str):
		DataStructure=dict()
		DataStructure["PointsY"]=self.WindField.Grid.Points[0]
		DataStructure["PointsZ"]=self.WindField.Grid.Points[1]
		DataStructure["Indices"]=self.WindField.Grid.Indices
		WindX=[]
		WindY=[]
		WindZ=[]
		for i in range(len(self.WindField.Wind)):
			WindX.append(self.WindField.Wind[i].WindValues[0,:])
			WindY.append(self.WindField.Wind[i].WindValues[1,:])
			WindZ.append(self.WindField.Wind[i].WindValues[2,:])
		LiDARMeasureParameters=[]
		LIDARMeasureValues=[]
		LIDARMeasureWindValues=[]
		for i in range(len(self.Measure)):
			Indices=self.Measure[i].Indices
			LiDARMeasureParameters.append(self.Measure[i].MeasureParameters)
			LIDARMeasureValues.append(self.GetLiDARMesure(i))
			LIDARMeasureWindValues.append(self.WindField.Wind[Indices[30]].WindValues[0,:])
			
		DataStructure["WindX"]=WindX
		DataStructure["WindY"]=WindY
		DataStructure["WindZ"]=WindZ
		DataStructure["LiDARMeasureParameters"]=LiDARMeasureParameters
		DataStructure["LiDARMeasureValues"]=LIDARMeasureValues
		DataStructure["LiDARMeasureWindValues"]=LIDARMeasureWindValues
		savemat(str, DataStructure)
	def GetLiDARMesure(self,Index):
		Indices=self.Measure[Index].Indices
		TimeVector=[0.0]*self.WindField.SimulationParameters.NSamples
		for i in range(self.WindField.SimulationParameters.NSamples):
			TimeVector[i]=i*self.WindField.SimulationParameters.SampleTime		
		DelayVector=[0.0]*len(Indices)
		for i in range(len(Indices)):
			DelayVector[i]=-float(self.Measure[Index].Distance[i])*cos(pi/180*self.Measure[Index].MeasureParameters[0])/self.WindField.Spectrum.WindMean
		#print(DelayVector) 
		MesureLiDAR=array([0.0]*len(TimeVector))
		for i in range(len(Indices)):
			x=self.WindField.Wind[Indices[i]].WindValues[0,:]
			y=self.WindField.Wind[Indices[i]].WindValues[1,:]
			z=self.WindField.Wind[Indices[i]].WindValues[2,:]
			f = interpolate.interp1d(TimeVector,x, kind='linear', axis=-1, copy=True, bounds_error=False, fill_value=self.WindField.Spectrum.WindMean)
			InterpolatedValuesX=f(array(TimeVector)-array([DelayVector[i]]*len(TimeVector)))
			fy = interpolate.interp1d(TimeVector,y, kind='linear', axis=-1, copy=True, bounds_error=False, fill_value=0)
			InterpolatedValuesY=fy(array(TimeVector)-array([DelayVector[i]]*len(TimeVector)))
			fz = interpolate.interp1d(TimeVector,z, kind='linear', axis=-1, copy=True, bounds_error=False, fill_value=0)
			InterpolatedValuesZ=fz(array(TimeVector)-array([DelayVector[i]]*len(TimeVector)))
			#print(sum(self.Measure[Index].Ponderation[i]))
			#print 'Phi=%s' % cos(float(self.Measure[Index].Phi)*pi/180)
			#print 'Theta=%s' % cos(float(self.Measure[Index].Theta)*pi/180)
			MesureLiDAR=MesureLiDAR+InterpolatedValuesX*self.Measure[Index].Ponderation[i]*cos(float(self.Measure[Index].Phi)*pi/180)
			MesureLiDAR=MesureLiDAR+InterpolatedValuesY*self.Measure[Index].Ponderation[i]*sin(float(self.Measure[Index].Phi)*pi/180)*cos(float(self.Measure[Index].Theta)*pi/180)
			MesureLiDAR=MesureLiDAR+InterpolatedValuesZ*self.Measure[Index].Ponderation[i]*sin(float(self.Measure[Index].Phi)*pi/180)*sin(float(self.Measure[Index].Theta)*pi/180)
		return MesureLiDAR
	def DisplayLiDARMesure(self,Index):
		Indices=self.Measure[Index].Indices
		TimeVector=[0.0]*self.WindField.SimulationParameters.NSamples
		for i in range(self.WindField.SimulationParameters.NSamples):
			TimeVector[i]=i*self.WindField.SimulationParameters.SampleTime		
		DelayVector=[0.0]*len(Indices)
		for i in range(len(Indices)):
			DelayVector[i]=-float(self.Measure[Index].Distance[i])*cos(pi/180*self.Measure[Index].MeasureParameters[0])/self.WindField.Spectrum.WindMean
		print(DelayVector)
		MesureLiDAR=array([0.0]*len(TimeVector))
		fig=plt.figure()
		fig.suptitle('Mesure LiDAR', fontsize=18)
		ax = fig.add_subplot(111)
		for i in range(len(Indices)):
			x=self.WindField.Wind[Indices[i]].WindValues[0,:]
			y=self.WindField.Wind[Indices[i]].WindValues[1,:]
			z=self.WindField.Wind[Indices[i]].WindValues[2,:]
			f = interpolate.interp1d(TimeVector,x, kind='linear', axis=-1, copy=True, bounds_error=False, fill_value=self.WindField.Spectrum.WindMean)
			InterpolatedValuesX=f(array(TimeVector)-array([DelayVector[i]]*len(TimeVector)))
			fy = interpolate.interp1d(TimeVector,y, kind='linear', axis=-1, copy=True, bounds_error=False, fill_value=0)
			InterpolatedValuesY=fy(array(TimeVector)-array([DelayVector[i]]*len(TimeVector)))
			fz = interpolate.interp1d(TimeVector,z, kind='linear', axis=-1, copy=True, bounds_error=False, fill_value=0)
			InterpolatedValuesZ=fz(array(TimeVector)-array([DelayVector[i]]*len(TimeVector)))
			print(sum(self.Measure[Index].Ponderation[i]))
			print('Phi=%s' % cos(float(self.Measure[Index].Phi)*pi/180))
			print('Theta=%s' % cos(float(self.Measure[Index].Theta)*pi/180))
			MesureLiDAR=MesureLiDAR+InterpolatedValuesX*self.Measure[Index].Ponderation[i]*cos(float(self.Measure[Index].Phi)*pi/180)
			MesureLiDAR=MesureLiDAR+InterpolatedValuesY*self.Measure[Index].Ponderation[i]*sin(float(self.Measure[Index].Phi)*pi/180)*cos(float(self.Measure[Index].Theta)*pi/180)
			MesureLiDAR=MesureLiDAR+InterpolatedValuesZ*self.Measure[Index].Ponderation[i]*sin(float(self.Measure[Index].Phi)*pi/180)*sin(float(self.Measure[Index].Theta)*pi/180)
			#ax.plot(TimeVector,MesureLiDAR)
		ax.plot(TimeVector,MesureLiDAR,'x',label='Mesure LiDAR')
		ax.plot(TimeVector,self.WindField.Wind[Indices[30]].WindValues[0,:],label='Vent Long')
		ax.set_xlabel('t(s)')
		ax.set_ylabel('Vent (m/s)')
		ax.legend()
		ax.grid()
		plt.show()	
		#essai de la fonction d'interpolation


if __name__ == "__main__":
	params = LiDARSimulationParameters(1000, 0.1)
	params.InitFromText("./data/Simulationcourte.li")
	spectrum = LiDARSpectrum()
	
	spectrum.InitSpectrumFromText("./data/Simulationcourte.li")
	spectrum.DisplaySpectrum()
	wind = LiDARWind(params)
	wind.SetWind(spectrum)
	wind.DisplayWind()