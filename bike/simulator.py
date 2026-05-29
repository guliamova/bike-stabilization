"""Run a single episode and collect statistics.

A goal is considered "reached" when the position is within `pos_tol`
of the origin (heading-agnostic). Final heading error is recorded
separately. The simulation continues for a short hold time after the
first reach to make sure the bike actually settles inside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .model import BikeState, BikeParams, step
from .controllers.base import Controller, wrap_angle


@dataclass
class Episode:
    name: str
    color: tuple[int, int, int]
    states:    list[tuple[float, float, float]] = field(default_factory=list)
    v_cmd:     list[float] = field(default_factory=list)
    delta_cmd: list[float] = field(default_factory=list)
    omega:     list[float] = field(default_factory=list)
    times:     list[float] = field(default_factory=list)

    path_length: float = 0.0
    mean_lin_speed: float = 0.0
    mean_ang_speed: float = 0.0
    time_to_goal: float = math.inf  # time to enter pos-tol ball
    success: bool = False           # ever entered the ball
    final_pos_err: float = 0.0
    final_head_err: float = 0.0


def run(controller: Controller, start: BikeState, params: BikeParams,
        dt: float = 0.02, t_max: float = 25.0,
        pos_tol: float = 0.3,
        hold_steps: int = 30) -> Episode:
    ep = Episode(name=controller.name, color=controller.color)
    state = BikeState(start.x, start.y, start.theta)
    t = 0.0
    held = 0
    n = int(t_max / dt)
    prev_xy = (state.x, state.y)
    reached_t = math.inf

    for _ in range(n):
        v, delta = controller.control(state)
        new_state = step(state, v, delta, dt, params)

        ep.states.append(state.as_tuple())
        ep.v_cmd.append(v)
        ep.delta_cmd.append(delta)
        ep.omega.append(v / params.wheelbase * math.tan(delta))
        ep.times.append(t)

        ep.path_length += math.hypot(new_state.x - prev_xy[0],
                                     new_state.y - prev_xy[1])
        prev_xy = (new_state.x, new_state.y)

        rho = math.hypot(new_state.x, new_state.y)
        if rho < pos_tol:
            if reached_t is math.inf:
                reached_t = t
            held += 1
            if held >= hold_steps:
                state = new_state
                t += dt
                ep.states.append(state.as_tuple())
                ep.times.append(t)
                break
        else:
            held = 0
            reached_t = math.inf  # had to re-enter

        state = new_state
        t += dt

    if ep.v_cmd:
        ep.mean_lin_speed = sum(abs(v) for v in ep.v_cmd) / len(ep.v_cmd)
        ep.mean_ang_speed = sum(abs(w) for w in ep.omega) / len(ep.omega)
    fs = ep.states[-1]
    ep.final_pos_err = math.hypot(fs[0], fs[1])
    ep.final_head_err = abs(wrap_angle(fs[2]))
    ep.success = ep.final_pos_err < pos_tol
    ep.time_to_goal = reached_t if reached_t is not math.inf else t_max
    return ep
