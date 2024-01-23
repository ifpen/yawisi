import os
import unittest
from yawisi.parameters import LiDARSimulationParameters
class TestSimulationParameters(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_params(self):
        filename = os.path.join(os.path.dirname(__file__), "config.ini")

        params = LiDARSimulationParameters(filename)
        print(params)

if __name__ == "__main__":
    unittest.main()
    
