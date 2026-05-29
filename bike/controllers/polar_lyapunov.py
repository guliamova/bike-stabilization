"""Smooth pose stabilization via Lyapunov control in polar coordinates.

Polar variables (goal at origin):
    rho   = sqrt(x^2 + y^2)
    alpha = atan2(-y, -x) - theta        (heading error to goal direction)
    beta  = -theta - alpha               (final-orientation error)

Control law (Corke "Robotics, Vision and Control", eq. 4.16):
    v     = k_rho * rho
    omega = k_alpha * alpha + k_beta * beta

For reverse motion: virtually rotate the bike by pi (alpha += pi, beta is
recomputed from the same heading definition), flip v sign.

Curvature kappa = omega / v stays finite as rho -> 0, so delta = atan(L*kappa)
keeps steering active for the final heading correction.
"""

from __future__ import annotations

import math

from .base import Controller, wrap_angle
from ..model import BikeState


class PolarLyapunov(Controller):
    name = "Polar Lyapunov"
    color = (255, 99, 132)  # warm red

    k_rho   = 1.0
    k_alpha = 6.0
    k_beta  = -1.5

    def control(self, state: BikeState) -> tuple[float, float]:
        x, y, th = state.x, state.y, state.theta
        rho = math.hypot(x, y)
        if rho < 1e-3 and abs(wrap_angle(th)) < math.radians(2):
            return 0.0, 0.0

        # alpha relative to "heading direction" (forward)
        alpha_fwd = wrap_angle(math.atan2(-y, -x) - th)
        if abs(alpha_fwd) > math.pi / 2:
            # reverse: think of bike facing the other way for alpha,
            # i.e. add pi to the apparent heading
            alpha = wrap_angle(alpha_fwd + math.pi)
            direction = -1.0
            # For beta, the "apparent heading" is th + pi
            beta = wrap_angle(-(th + math.pi) - alpha)
        else:
            alpha = alpha_fwd
            direction = 1.0
            beta = wrap_angle(-th - alpha)

        v_mag = self.k_rho * rho
        omega = self.k_alpha * alpha + self.k_beta * beta
        kappa = omega / max(v_mag, 1e-3)

        v = direction * min(v_mag, self.params.v_max)
        delta = math.atan(self.params.wheelbase * kappa)
        if direction < 0:
            delta = -delta
        delta = max(-self.params.delta_max, min(self.params.delta_max, delta))
        return v, delta
