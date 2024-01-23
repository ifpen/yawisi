import os
import unittest
from yawisi.parameters import LiDARSimulationParameters
from yawisi.spectrum import LiDARSpectrum
from yawisi.display import display_spectrum


class TestSpectrum(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_params(self):
        filename = os.path.join(os.path.dirname(__file__), "config.ini")

        params = LiDARSimulationParameters(filename)
        params.n_samples = 1000
        params.sample_time = 1
        spectrum = LiDARSpectrum(params)
      
        display_spectrum(spectrum)


if __name__ == "__main__":
    unittest.main()