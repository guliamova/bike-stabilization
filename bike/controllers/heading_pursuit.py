"""Naive Heading-Pursuit baseline.

Drives toward the origin by simple proportional heading feedback. It does
NOT fix the final orientation -- a useful baseline to compare against
controllers that respect theta=0 at the goal. Stops when within position
tolerance regardless of heading.
"""

from __future__ import annotations

import math

from .base import Controller, wrap_angle
from ..model import BikeState


class HeadingPursuit(Controller):
    name = "Heading Pursuit"
    color = (75, 192, 120)  # green

    k_head = 2.5
    v_cruise = 1.5

    def control(self, state: BikeState) -> tuple[float, float]:
        x, y, th = state.x, state.y, state.theta
        rho = math.hypot(x, y)
        if rho < 0.12:
            return 0.0, 0.0   # stop at position; final theta is whatever

        desired_th = math.atan2(-y, -x)
        h_err = wrap_angle(desired_th - th)
        reverse = abs(h_err) > math.pi / 2
        if reverse:
            h_err = wrap_angle(h_err + math.pi)
            v = -self.v_cruise * min(1.0, rho)
            delta = -self.k_head * h_err
        else:
            v = self.v_cruise * min(1.0, rho)
            delta = self.k_head * h_err
        delta = max(-self.params.delta_max, min(self.params.delta_max, delta))
        return v, delta
