from typing import Optional, Dict
import numpy as np
from boundary.boundary_conditions import Sides, BoundaryConditions


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BoundaryConditions1D(BoundaryConditions):
    sides = [Sides.LEFT, Sides.RIGHT]

    def __init__(self, boundary_conditions: Optional[Dict[str, np.ndarray]] = None):
        super().__init__(boundary_conditions)

    def restrict(self) -> BoundaryConditions:
        values_restricted = {side: bc.copy() for side, bc in self.values.items()}
        return BoundaryConditions1D(values_restricted)

