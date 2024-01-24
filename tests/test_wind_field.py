import os
import unittest
from yawisi.parameters import SimulationParameters
from yawisi.wind_field import WindField
from yawisi.locations import Locations
from yawisi.display import display_coherence_function, display_field
from yawisi.io import to_bts
import matplotlib.pyplot as plt

class TestWindField(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_points_wind_field(self):
        filename = os.path.join(os.path.dirname(__file__), "config.ini")
        params = SimulationParameters(filename)
        params.n_samples = 2000
        params.sample_time = 0.1

        wind_field = WindField(params)
        pts = [
            (0, 0),
            (0, 1),
            (1, 0)
        ]
        wind_field.locations = Locations.create("points")
        wind_field.locations.add_points(pts)
        
        freq, coherence = wind_field.get_coherence_function()
        display_coherence_function(freq, coherence)

        print(wind_field._get_coherence_matrix(1, wind_field.locations.get_distance_matrix()))

        wind_field.compute()
        display_field(wind_field=wind_field)

    def test_grid_field(self):
        filename = os.path.join(os.path.dirname(__file__), "config.ini")
        params = SimulationParameters(filename)
        params.n_samples = 2000
        params.sample_time = 0.1

        wind_field = WindField(params)
        wind_field.compute()
        #display_field(wind_field=wind_field)

        ts = wind_field.get_uvwt()
        print(ts.shape)
        for i in range(3):
            plt.plot(ts[i, 6 , 6, :])
        plt.show()

        uhub = wind_field.get_umean()
        print(uhub)

    def test_to_bts(self):
        filename = os.path.join(os.path.dirname(__file__), "config.ini")
        
        params = SimulationParameters(filename)
        params.n_samples = 2000
        params.sample_time = 0.1

        wind_field = WindField(params)
        wind_field.compute()
        to_bts(wind_field=wind_field, path="./test.bts")


if __name__ == "__main__":
    unittest.main()
