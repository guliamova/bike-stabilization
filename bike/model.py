"""Kinematic bicycle model.

State:  q = [x, y, theta]
Input:  u = [v, delta]      (linear velocity, steering angle)
Dyn:    x_dot     = v * cos(theta)
        y_dot     = v * sin(theta)
        theta_dot = v / L * tan(delta)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BikeParams:
    wheelbase: float = 1.0          # L, distance between axles [m]
    v_max: float = 2.0              # max |v|        [m/s]
    delta_max: float = math.radians(35.0)  # max |steering| [rad]
    allow_reverse: bool = True      # allow v < 0


@dataclass
class BikeState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # heading [rad]

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.theta


def step(state: BikeState, v: float, delta: float, dt: float,
         params: BikeParams) -> BikeState:
    """Integrate the bicycle one step with RK4 (kinematic, no slip)."""
    # saturate inputs
    v = max(-params.v_max if params.allow_reverse else 0.0,
            min(params.v_max, v))
    delta = max(-params.delta_max, min(params.delta_max, delta))

    def f(x, y, th):
        return (v * math.cos(th),
                v * math.sin(th),
                v / params.wheelbase * math.tan(delta))

    x, y, th = state.x, state.y, state.theta
    k1 = f(x,                y,                th)
    k2 = f(x + 0.5*dt*k1[0], y + 0.5*dt*k1[1], th + 0.5*dt*k1[2])
    k3 = f(x + 0.5*dt*k2[0], y + 0.5*dt*k2[1], th + 0.5*dt*k2[2])
    k4 = f(x +     dt*k3[0], y +     dt*k3[1], th +     dt*k3[2])

    nx  = x  + dt/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    ny  = y  + dt/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    nth = th + dt/6 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    nth = math.atan2(math.sin(nth), math.cos(nth))  # wrap
    return BikeState(nx, ny, nth)
