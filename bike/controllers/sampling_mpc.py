from __future__ import annotations

import math
import numpy as np

from .base import Controller, wrap_angle
from ..model import BikeState


class SamplingMPC(Controller):
    name = "Sampling MPC"
    color = (54, 162, 235)

    H1 = 8
    H2 = 32
    H1_near = 4
    H2_near = 18
    near_radius = 1.0
    park_radius = 0.30
    sim_dt = 0.06

    w_pos      = 8.0
    w_ang      = 1.0
    w_path_pos = 0.20
    w_u        = 0.0002

    _vmax_frac = np.array([-1.0, -0.7, -0.5, -0.35,
                            0.35, 0.5, 0.7, 1.0])
    _n_delta   = 13
    K_keep     = 12

    def __init__(self, params):
        super().__init__(params)
        self._L  = params.wheelbase
        vmax     = params.v_max
        dmax     = params.delta_max
        self._vs = self._vmax_frac * vmax
        self._ds = dmax * np.linspace(-1, 1, self._n_delta)
        V, D     = np.meshgrid(self._vs, self._ds, indexing="ij")
        self._V  = V.flatten()
        self._D  = D.flatten()
        self._omega = self._V / self._L * np.tan(self._D)
        self._u_cost = self.w_u * (self._V**2 + self._D**2)

    def _rollout(self, xs, ys, ths, V, D, omega, n_steps):
        xs = xs.copy(); ys = ys.copy(); ths = ths.copy()
        running = np.zeros_like(xs)
        dt = self.sim_dt
        for _ in range(n_steps):
            xs  += dt * V * np.cos(ths)
            ys  += dt * V * np.sin(ths)
            ths += dt * omega
            running += self.w_path_pos * np.sqrt(xs*xs + ys*ys)
        return xs, ys, ths, running

    def _terminal(self, xs, ys, ths):
        ths = np.arctan2(np.sin(ths), np.cos(ths))
        return (self.w_pos * np.sqrt(xs*xs + ys*ys)
                + self.w_ang * ths*ths)

    def _park(self, x, y, th):
        """Analytic endgame: head toward origin, slow with ρ."""
        rho = math.hypot(x, y)
        if rho < 0.04:
            return 0.0, 0.0
        # forward / reverse: face the origin
        ang_to_goal = wrap_angle(math.atan2(-y, -x) - th)
        if abs(ang_to_goal) > math.pi / 2:
            ang_to_goal = wrap_angle(ang_to_goal + math.pi)
            v = -self.params.v_max * 0.35 * min(1.0, rho / 0.3)
            delta = -2.5 * ang_to_goal
        else:
            v = self.params.v_max * 0.35 * min(1.0, rho / 0.3)
            delta = 2.5 * ang_to_goal
        delta = max(-self.params.delta_max, min(self.params.delta_max, delta))
        return v, delta

    def control(self, state: BikeState) -> tuple[float, float]:
        x0, y0, th0 = state.x, state.y, state.theta
        rho0 = math.hypot(x0, y0)

        if rho0 < self.park_radius:
            return self._park(x0, y0, th0)

        if rho0 < self.near_radius:
            H1, H2 = self.H1_near, self.H2_near
        else:
            H1, H2 = self.H1, self.H2

        M = self._V.shape[0]

        # Stage A
        xs1 = np.full(M, x0, dtype=float)
        ys1 = np.full(M, y0, dtype=float)
        ths1 = np.full(M, th0, dtype=float)
        xs1, ys1, ths1, run1 = self._rollout(
            xs1, ys1, ths1, self._V, self._D, self._omega, H1)

        xsG, ysG, thsG, runG = self._rollout(
            xs1, ys1, ths1, self._V, self._D, self._omega, H2)
        greedy = self._terminal(xsG, ysG, thsG) + run1 + runG + self._u_cost
        topk = np.argpartition(greedy, self.K_keep)[:self.K_keep]

        # Stage B: K × M
        K = self.K_keep
        x_s = xs1[topk]; y_s = ys1[topk]; th_s = ths1[topk]
        run_s = run1[topk]; uc_a = self._u_cost[topk]
        a1_idx = topk

        xs2 = np.repeat(x_s, M); ys2 = np.repeat(y_s, M); ths2 = np.repeat(th_s, M)
        run2_init = np.repeat(run_s, M)
        uc_a_rep  = np.repeat(uc_a, M)
        a1_rep    = np.repeat(a1_idx, M)
        V2 = np.tile(self._V, K); D2 = np.tile(self._D, K); om2 = np.tile(self._omega, K)
        uc_b = np.tile(self._u_cost, K)

        xs2, ys2, ths2, run2 = self._rollout(xs2, ys2, ths2, V2, D2, om2, H2)
        cost = (self._terminal(xs2, ys2, ths2)
                + run2_init + run2 + uc_a_rep + uc_b)

        idx = int(np.argmin(cost))
        best_a1 = int(a1_rep[idx])
        return float(self._V[best_a1]), float(self._D[best_a1])
