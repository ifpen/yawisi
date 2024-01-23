import numpy as np

from yawisi.parameters import LiDARSimulationParameters
from yawisi.kernels import Kaimal, Karman

class LiDARSpectrum:

    def __init__(self, params:LiDARSimulationParameters):
        self.params = params
        self.kernel = Kaimal(params)
        
    def compute(self, N, Ts):
        FMax = 1./ Ts
        freq = np.arange(0, FMax, FMax/N)
        array = np.zeros(shape=(N, 3))
        array[:, 0] = self.kernel(0, freq)
        array[:, 1] = self.kernel(1, freq)
        array[:, 2] = self.kernel(2, freq)

        return freq, array 
