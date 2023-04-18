import cv2
from typing import Optional, Dict
import numpy as np
from boundary.boundary_conditions import Sides, BoundaryConditions


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BoundaryConditions2D(BoundaryConditions):
    sides = [Sides.LEFT, Sides.RIGHT, Sides.TOP, Sides.BOTTOM]

    def __init__(self, boundary_conditions: Optional[Dict[str, np.ndarray]] = None):
        super().__init__(boundary_conditions)

    def restrict_side(self, bc: np.ndarray):
        bc_restricted_size = len(bc) // 2
        bc_restricted = cv2.resize(bc, (1, bc_restricted_size), interpolation=cv2.INTER_AREA)
        return bc_restricted.reshape(bc_restricted_size)

    def restrict(self) -> BoundaryConditions:
        values_restricted = {side: self.restrict_side(bc) for side, bc in self.values.items()}
        return BoundaryConditions2D(values_restricted)

