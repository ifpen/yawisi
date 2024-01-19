

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
