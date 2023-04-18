from typing import Optional, Dict
import numpy as np
from abc import ABC, abstractmethod
from common import Sides


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

