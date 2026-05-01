import math as m
import pytensor.tensor as pt
from numpy import pi as PI, inf, where

class NewPrior:
    def __init__(self):
        raise NotImplementedError

    def logpdf(self, v):
        raise NotImplementedError
    
class NewUniformPrior(NewPrior):
    def __init__(self, a: float, b: float):
        self.a, self.b = a, b
        self.lnc = -m.log(b-a)

    def logpdf(self, v):
        inbounds = pt.and_(self.a < v, v < self.b)
        return where(inbounds, self.lnc, -inf)

    def __str__(self):
        return f'U(a = {self.a}, b = {self.b})'

    def __repr__(self):
        return f'UniformPrior({self.a}, {self.b})'
    
class NewNormalPrior(NewPrior):
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)
        self._f1 = 1 / m.sqrt(2*PI*std**2)
        self._lf1 = m.log(self._f1)
        self._f2 = 1 / (2*std**2)

    def logpdf(self, x):
        return self._lf1 - self._f2*(x - self.mean)**2

    def __str__(self):
        return f'N(μ = {self.mean}, σ = {self.std})'

    def __repr__(self):
        return f'NormalPrior({self.mean}, {self.std})'