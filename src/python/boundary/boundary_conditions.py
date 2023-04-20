from typing import Dict
import numpy as np
from abc import ABC, abstractmethod
from common import Sides


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class BoundaryConditions(ABC):
    sides = [Sides.LEFT, Sides.RIGHT]

    def __init__(self, boundary_conditions: Dict[str, np.ndarray]):
        self.values = boundary_conditions

    @abstractmethod
    def restrict(self) -> 'BoundaryConditions':
        pass

    def get_side(self, side: Sides) ->np.ndarray:
        return self.values[side]

    def __sub__(self, other: 'BoundaryConditions') -> 'BoundaryConditions':
        boundary_conditions = {side: bc - other.get_side(side) for side, bc in self.values.items()}
        return self.__class__(boundary_conditions)

