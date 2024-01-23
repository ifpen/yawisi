import numpy as np

from yawisi.parameters import LiDARSimulationParameters
from yawisi.kernels import Kaimal, Karman

class LiDARSpectrum:

    def __init__(self, params:LiDARSimulationParameters):
        self.params = params
        try:
            kind = self.params.kind.lower()
            self.kernel = {
                "kaimal": Kaimal,
                "karman": Karman
            }[kind](params)
        except KeyError as er:
            raise KeyError(f"spectrum {kind} unknown (only kaimal, karman)")
        
        self.freq, self.array = self._compute(params.n_samples, params.sample_time)

    def symetrized(self, i):
        return self.array[:, i] + self.array[:, i][::-1]
        
    def _compute(self, N, Ts):
        FMax = 1./ Ts
        freq = np.arange(0, FMax, FMax/N)
        array = np.zeros(shape=(N, 3))
        for i in range(3):
            array[:, i] = self.kernel(i, freq)

        return freq, array 
