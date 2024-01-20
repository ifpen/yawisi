import math
from typing import Any

class Kernel:
	def __init__(self, spectrum) -> None:
		self.spectrum = spectrum

	def __call__(self, sigma, freq, *args, **kwds):
		pass


class Kaimal(Kernel):
	def __init__(self, spectrum) -> None:
		super().__init__(spectrum)

	def __call__(self, sigma, freq, *args, **kwds):
		Kv = 0.475*(sigma**2)*self.spectrum.Lv/self.spectrum.WindMean
		Tv = self.spectrum.Lv/self.spectrum.WindMean
		return Kv/(1+2*math.pi*freq*Tv)**(5/3)


class CoherenceKernel:

	def __init__(self) -> None:
		pass

	def __call__(self, freq, *args: Any, **kwds: Any) -> Any:
		return ((2*math.pi*freq/100)**2 + 0.003**2)**(1./2)

	

