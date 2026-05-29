"""Common controller interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

from ..model import BikeState, BikeParams


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Controller(ABC):
    name: str = "base"
    color: tuple[int, int, int] = (200, 200, 200)

    def __init__(self, params: BikeParams):
        self.params = params

    @abstractmethod
    def control(self, state: BikeState) -> tuple[float, float]:
        """Return (v, delta) that drives state -> origin (0, 0, 0)."""

    def at_goal(self, state: BikeState,
                pos_tol: float = 0.05, ang_tol: float = math.radians(3.0)
                ) -> bool:
        rho = math.hypot(state.x, state.y)
        return rho < pos_tol and abs(wrap_angle(state.theta)) < ang_tol
