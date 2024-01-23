import os
import unittest
from yawisi.parameters import LiDARSimulationParameters
from yawisi.wind import LiDARWind
from yawisi.display import display_wind
import matplotlib.pyplot as plt

class TestWind(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_wind(self):
        filename = os.path.join(os.path.dirname(__file__), "../data", "Simulationcourte.li")
        params = LiDARSimulationParameters(filename)
        print(params)

        wind = LiDARWind(params)
        wind.compute()

        display_wind(wind)

        

if __name__ == "__main__":
    unittest.main()
