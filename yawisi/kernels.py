import math
from yawisi.parameters import LiDARSimulationParameters


class Kernel:
	def __init__(self, params: LiDARSimulationParameters):
		self.Kv = [
			0.475*(params.sigma_x**2)*params.Lv/params.wind_mean,
			0.475*(params.sigma_y**2)*params.Lv/params.wind_mean,
			0.475*(params.sigma_z**2)*params.Lv/params.wind_mean
		]
		self.Tv = params.Lv/params.wind_mean

	def __call__(self, sigma, freq, *args, **kwds):
		pass


class Kaimal(Kernel):
	def __init__(self, params: LiDARSimulationParameters):
		super().__init__(params)

	def __call__(self, i_sigma, freq, *args, **kwds):		
		return self.Kv[i_sigma]/(1.+(2*math.pi*freq)*self.Tv)**(5/3)
	


class Karman(Kernel):
	def __init__(self, params: LiDARSimulationParameters):
		super().__init__(params)

	def __call__(self, i_sigma, freq, *args, **kwds):
		return self.Kv[i_sigma]/(1.+((2*math.pi*freq)*self.Tv)**2)**(5/6)


class CoherenceKernel:

	def __init__(self) -> None:
		pass

	def __call__(self, freq, *args, **kwds):
		return ((2*math.pi*freq/100)**2 + 0.003**2)**(1./2)

	

