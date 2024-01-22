import os
import unittest
from pywisim.parameters import LiDARSimulationParameters
from pywisim.wind import LiDARWind
from pywisim.display import display_wind
import matplotlib.pyplot as plt

class TestWind(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_wind(self):
        filename = os.path.join(os.path.dirname(__file__), "../data", "Simulationcourte.li")
        params = LiDARSimulationParameters(filename)

        wind = LiDARWind(params)
        wind.compute()

        display_wind(wind)

        

if __name__ == "__main__":
    unittest.main()
