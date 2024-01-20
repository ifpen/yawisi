import unittest
from wgen.locations import Grid
from wgen.display import display_points
import matplotlib.pyplot as plt

class TestSimulationParameters(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_grid(self):
        grid = Grid(100, 80, 10, 10)
        display_points(grid)

        distance_map = grid.get_distance_matrix()
        print(distance_map.shape)

if __name__ == "__main__":
    unittest.main()
   