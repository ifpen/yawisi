import os
import unittest
from wgen.parameters import LiDARSimulationParameters
from wgen.wind_field import LiDARWindField
from wgen.display import display_coherence_function
import matplotlib.pyplot as plt

class TestWindField(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_grid(self):
        filename = os.path.join(os.path.dirname(__file__), "../data", "Simulationcourte.li")
        params = LiDARSimulationParameters(filename)

        wind_field = LiDARWindField(params)

        freq, coherence = wind_field.get_coherence_function()
        display_coherence_function(freq, coherence)

        

if __name__ == "__main__":
    unittest.main()
