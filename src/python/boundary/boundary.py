import numpy as np


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Boundary:
    def __init__(self, ghost: np.ndarray, inside: np.ndarray, h: float):
        self.ghost = ghost
        self.inside = inside
        self.h = h

    def get_derivative(self):
        return (self.ghost - self.inside) / self.h

    def set_derivative(self, boundary_conditions: np.ndarray):
        self.ghost[:] = boundary_conditions * self.h + self.inside

