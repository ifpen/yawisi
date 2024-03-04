import math
from yawisi.parameters import SimulationParameters


class Kernel:
    def __init__(self, params: SimulationParameters):
        self.Lk = [params.scale_1, params.scale_2, params.scale_3]
        self.var_k = [params.sigma_1**2, params.sigma_2**2, params.sigma_3**2]
        self.Vhub = params.wind_mean

    def __call__(self, freq, k):
        pass


class Kaimal(Kernel):
    def __init__(self, params: SimulationParameters):
        super().__init__(params)

    def __call__(self, freq, k):
        numerator = 4 * self.var_k[k] * self.Lk[k] / self.Vhub
        denominator = (1 + 6 * freq * self.Lk[k] / self.Vhub) ** (5 / 3)
        return numerator / denominator


class Karman(Kernel):
    def __init__(self, params: SimulationParameters):
        super().__init__(params)

    def __call__(self, freq, k):
        fr = self.Lk[k] * freq / self.Vhub
        if k == 0:
            S = 4 * fr / (1 + 71 * fr**2) ** (5 / 6)
        else:
            S = 4 * fr * (1 + 755 * fr**2) / (1 + 283 * fr**2) ** (11 / 6)
        return S * self.var_k[k] / freq


class CoherenceKernel:
    def __init__(self) -> None:
        pass

    def __call__(self, freq, *args, **kwds):
        return ((2 * math.pi * freq / 100) ** 2 + 0.003**2) ** (1.0 / 2)
