import numpy as np
from boundary.boundary_conditions import BoundaryConditions


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class UpscalingParams:
    def __init__(self, f: np.ndarray, integral_constraints: np.ndarray, boundary_conditions: BoundaryConditions):
        self.f = f
        self.integral_constraints = integral_constraints
        self.boundary_conditions = boundary_conditions
