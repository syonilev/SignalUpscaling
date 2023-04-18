import cv2
from typing import Optional, Dict
import numpy as np
from scipy.signal import convolve2d
from abc import ABC, abstractmethod
from enum import Enum


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class Sides(Enum):
    LEFT, RIGHT, TOP, BOTTOM, BACK, FRONT = range(6)


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BoundaryConditions(ABC):
    sides = [Sides.LEFT, Sides.RIGHT]

    def __init__(self, boundary_conditions: Optional[Dict[str, np.ndarray]] = None):
        self.values = self.boundary_conditions_default if boundary_conditions is None else boundary_conditions

    @property
    def boundary_conditions_default(self):
        return {side: None for side in self.sides}

    @abstractmethod
    def restrict(self) -> 'BoundaryConditions':
        pass

    def get_side(self, side: Sides):
        return self.values[side]

    def __sub__(self, other: 'BoundaryConditions') -> 'BoundaryConditions':
        boundary_conditions = {side: bc - other.get_side(side) for side, bc in self.values.items()}
        return self.__class__(boundary_conditions)



# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BoundaryConditions1D(BoundaryConditions):
    sides = [Sides.LEFT, Sides.RIGHT]

    def __init__(self, boundary_conditions: Optional[Dict[str, np.ndarray]] = None):
        super().__init__(boundary_conditions)

    def restrict(self) -> BoundaryConditions:
        raise "!!! NOT IMPLEMENTED !!!"


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


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BoundaryConditions3D(BoundaryConditions):
    sides = [Sides.LEFT, Sides.RIGHT, Sides.TOP, Sides.BOTTOM, Sides.BACK, Sides.FRONT]

    def __init__(self, boundary_conditions: Optional[Dict[str, np.ndarray]] = None):
        super().__init__(boundary_conditions)

    def restrict(self) -> BoundaryConditions:
        raise "!!! NOT IMPLEMENTED !!!"