import os
import unittest
from yawisi.parameters import SimulationParameters
from yawisi.spectrum import Spectrum
from yawisi.display import display_spectrum

import numpy as np
import matplotlib.pyplot as plt


class TestSpectrum(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_params(self):
        filename = os.path.join(os.path.dirname(__file__), "config.ini")

        params = SimulationParameters(filename)
        params.n_samples = 1024
        params.sample_time = 0.05
        spectrum = Spectrum(params)

        freq, array = spectrum.freq, spectrum.array

        df = freq[1] - freq[0]

        print(array.shape)
        print(np.sqrt(df * np.sum(array, axis=0)))

        display_spectrum(spectrum)

        plt.plot(spectrum.symetrized(0))
        plt.show()


if __name__ == "__main__":
    unittest.main()
