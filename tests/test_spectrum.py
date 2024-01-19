import os
import unittest
from wgen.parameters import LiDARSimulationParameters
from wgen.spectrum import LiDARSpectrum
from wgen.display import display_spectrum


class TestSpectrum(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_params(self):
        filename = os.path.join(os.path.dirname(__file__), "../data", "Simulationcourte.li")

        params = LiDARSimulationParameters(filename)
        spectrum = LiDARSpectrum(params)

        display_spectrum(spectrum)


if __name__ == "__main__":
    unittest.main()