"""Sampling MPC (vectorized with NumPy).

At each step, we roll out all (v, delta) candidates in parallel using
the kinematic bicycle in Euler form. Cost is integrated over the horizon
and the candidate with smallest cost is picked.
"""

from __future__ import annotations

import math
import numpy as np

from .base import Controller, wrap_angle
from ..model import BikeState


class SamplingMPC(Controller):
    name = "Sampling MPC"
    color = (54, 162, 235)

    horizon_steps = 20
    horizon_steps_near = 18
    sim_dt = 0.06
    w_pos = 3.0
    w_ang = 0.5
    w_path_pos = 0.05
    w_u = 0.002

    def __init__(self, params):
        super().__init__(params)
        vmax = params.v_max
        v_fracs = np.array([-1.0, -0.6, -0.3, -0.1, -0.03,
                            0.0, 0.03, 0.1, 0.3, 0.6, 1.0])
        self._vs = v_fracs * vmax
        n_delta = 11
        self._ds = params.delta_max * np.linspace(-1, 1, n_delta)

        V, D = np.meshgrid(self._vs, self._ds, indexing="ij")
        self._V = V.flatten()        # (M,)
        self._D = D.flatten()
        self._tan_D = np.tan(self._D)
        self._u_cost = self.w_u * (self._V**2 + self._D**2)
        self._L = params.wheelbase

    def control(self, state: BikeState) -> tuple[float, float]:
        x0, y0, th0 = state.x, state.y, state.theta
        if (math.hypot(x0, y0) < 0.05
                and abs(wrap_angle(th0)) < math.radians(4)):
            return 0.0, 0.0

        horizon = (self.horizon_steps_near
                   if math.hypot(x0, y0) < 0.7
                   else self.horizon_steps)

        M = self._V.shape[0]
        xs = np.full(M, x0, dtype=float)
        ys = np.full(M, y0, dtype=float)
        ths = np.full(M, th0, dtype=float)

        running = np.zeros(M)
        dt = self.sim_dt
        omega_term = self._V / self._L * self._tan_D
        for _ in range(horizon):
            xs  += dt * self._V * np.cos(ths)
            ys  += dt * self._V * np.sin(ths)
            ths += dt * omega_term
            running += self.w_path_pos * (xs*xs + ys*ys)

        # wrap thetas
        ths = np.arctan2(np.sin(ths), np.cos(ths))
        terminal = self.w_pos * (xs*xs + ys*ys) + self.w_ang * ths*ths
        cost = terminal + running + self._u_cost

        idx = int(np.argmin(cost))
        return float(self._V[idx]), float(self._D[idx])
