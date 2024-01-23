import os
import unittest
from yawisi.parameters import LiDARSimulationParameters
class TestSimulationParameters(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_params(self):
        filename = os.path.join(os.path.dirname(__file__), "../data", "Simulationcourte.li")

        params = LiDARSimulationParameters(filename)
        print(params)

if __name__ == "__main__":
    unittest.main()
    # params = LiDARSimulationParameters.init_from_text("Simulationcourte.li")
    # print(params)

    # spectrum = LiDARSpectrum.init_spectrum_from_text("Simulationcourte.li")
    # print(spectrum)

    # spectrum.display()

    # spectrum.GenerateSignal(params)
    
