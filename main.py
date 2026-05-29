"""Pygame animation of bicycle stabilization controllers.

Controls:
    SPACE       reset with a new random start
    R           reset with the same start
    1 / 2 / 3   choose start preset
    TAB         pause / resume
    Q / ESC     quit
"""

from __future__ import annotations

import math
import random
import sys

import pygame

from bike import BikeState, BikeParams, step
from bike.controllers import ALL_CONTROLLERS


# ---------- visuals ----------
W, H = 1280, 760
WORLD_X_MIN, WORLD_X_MAX = -6, 6
WORLD_Y_MIN, WORLD_Y_MAX = -3.6, 3.6   # 6/3.6 = 1280/760 approx

BG_TOP     = (32, 34, 42)
BG_BOTTOM  = (18, 19, 25)
GRID       = (60, 64, 76)
GRID_AXIS  = (110, 116, 130)
FG         = (235, 236, 240)
SUB        = (170, 175, 188)
GOAL       = (255, 215, 90)


def world_to_screen(x: float, y: float) -> tuple[int, int]:
    sx = (x - WORLD_X_MIN) / (WORLD_X_MAX - WORLD_X_MIN) * W
    sy = H - (y - WORLD_Y_MIN) / (WORLD_Y_MAX - WORLD_Y_MIN) * H
    return int(sx), int(sy)


def draw_grid(surf: pygame.Surface) -> None:
    # vertical gradient background
    for i in range(H):
        t = i / H
        r = int(BG_TOP[0] * (1-t) + BG_BOTTOM[0] * t)
        g = int(BG_TOP[1] * (1-t) + BG_BOTTOM[1] * t)
        b = int(BG_TOP[2] * (1-t) + BG_BOTTOM[2] * t)
        pygame.draw.line(surf, (r, g, b), (0, i), (W, i))

    # grid lines every 1m
    for ix in range(int(WORLD_X_MIN), int(WORLD_X_MAX) + 1):
        x0, _ = world_to_screen(ix, 0)
        color = GRID_AXIS if ix == 0 else GRID
        pygame.draw.line(surf, color, (x0, 0), (x0, H), 1 if ix != 0 else 2)
    for iy in range(int(WORLD_Y_MIN), int(WORLD_Y_MAX) + 1):
        _, y0 = world_to_screen(0, iy)
        color = GRID_AXIS if iy == 0 else GRID
        pygame.draw.line(surf, color, (0, y0), (W, y0), 1 if iy != 0 else 2)


def draw_goal(surf: pygame.Surface) -> None:
    gx, gy = world_to_screen(0, 0)
    # heading arrow at goal pointing +x
    end_x = gx + 38
    pygame.draw.circle(surf, GOAL, (gx, gy), 10, 2)
    pygame.draw.line(surf, GOAL, (gx, gy), (end_x, gy), 3)
    pygame.draw.polygon(surf, GOAL,
                        [(end_x, gy), (end_x - 8, gy - 5), (end_x - 8, gy + 5)])


def draw_bike(surf: pygame.Surface, state: BikeState,
              params: BikeParams, color: tuple[int, int, int],
              alpha: int = 255, body_scale: float = 1.0) -> None:
    """Draw a top-down bicycle: two wheels connected by a frame."""
    # convert meters to pixels (use x-scale)
    px_per_m = W / (WORLD_X_MAX - WORLD_X_MIN)
    L_px = params.wheelbase * px_per_m * body_scale

    cx, cy = world_to_screen(state.x, state.y)
    c, s = math.cos(state.theta), -math.sin(state.theta)  # screen-y is flipped

    # rear axle at (state.x, state.y), front axle at distance L forward
    front = (cx + c * L_px, cy + s * L_px)
    rear  = (cx, cy)

    # frame
    pygame.draw.line(surf, color, rear, front, 4)

    wheel_len = 14 * body_scale
    for (wx, wy) in (rear, front):
        # wheel oriented along bike heading
        x1 = wx - c * wheel_len/2
        y1 = wy - s * wheel_len/2
        x2 = wx + c * wheel_len/2
        y2 = wy + s * wheel_len/2
        pygame.draw.line(surf, color, (x1, y1), (x2, y2), 5)

    # rider dot
    pygame.draw.circle(surf, color, (int(cx + c*L_px*0.4), int(cy + s*L_px*0.4)), 3)


def draw_trail(surf: pygame.Surface, trail: list[tuple[float,float]],
               color: tuple[int,int,int]) -> None:
    if len(trail) < 2:
        return
    pts = [world_to_screen(x, y) for (x, y) in trail]
    pygame.draw.lines(surf, color, False, pts, 2)


# ---------- main ----------
def random_start(rng) -> BikeState:
    return BikeState(rng.uniform(-4.5, 4.5),
                     rng.uniform(-3, 3),
                     rng.uniform(-math.pi, math.pi))


PRESETS = [
    BikeState(-4.0, 2.5, math.radians(-30)),
    BikeState( 3.5, -2.2, math.radians(150)),
    BikeState(-3.0, -2.5, math.radians(60)),
]


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Bicycle stabilization to (0, 0, 0)")
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("georgia", 22, bold=True)
    font  = pygame.font.SysFont("consolas", 16)
    small = pygame.font.SysFont("consolas", 13)

    rng = random.Random()
    rng.seed()

    params = BikeParams()
    start = PRESETS[0]
    controllers = [C(params) for C in ALL_CONTROLLERS]
    states = [BikeState(start.x, start.y, start.theta) for _ in controllers]
    trails: list[list[tuple[float,float]]] = [[(s.x, s.y)] for s in states]
    times = [0.0 for _ in controllers]
    arrived = [False for _ in controllers]
    metrics = [{"len": 0.0, "v": [], "w": []} for _ in controllers]

    paused = False
    sim_dt = 0.02
    speed = 1.0   # sim time per real second
    running = True

    def reset(new_start: BikeState | None = None):
        nonlocal start, controllers, states, trails, times, arrived, metrics
        if new_start is not None:
            start = new_start
        controllers = [C(params) for C in ALL_CONTROLLERS]
        states = [BikeState(start.x, start.y, start.theta) for _ in controllers]
        trails = [[(s.x, s.y)] for s in states]
        times = [0.0 for _ in controllers]
        arrived = [False for _ in controllers]
        metrics = [{"len": 0.0, "v": [], "w": []} for _ in controllers]

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    reset(random_start(rng))
                elif event.key == pygame.K_r:
                    reset(start)
                elif event.key == pygame.K_TAB:
                    paused = not paused
                elif event.key == pygame.K_1:
                    reset(PRESETS[0])
                elif event.key == pygame.K_2:
                    reset(PRESETS[1])
                elif event.key == pygame.K_3:
                    reset(PRESETS[2])
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(4.0, speed * 1.5)
                elif event.key == pygame.K_MINUS:
                    speed = max(0.25, speed / 1.5)

        if not paused:
            steps = max(1, int(speed))
            for _ in range(steps):
                for i, ctrl in enumerate(controllers):
                    if arrived[i]:
                        continue
                    v, d = ctrl.control(states[i])
                    new_state = step(states[i], v, d, sim_dt, params)

                    metrics[i]["len"] += math.hypot(new_state.x - states[i].x,
                                                    new_state.y - states[i].y)
                    metrics[i]["v"].append(abs(v))
                    metrics[i]["w"].append(abs(v / params.wheelbase * math.tan(d)))

                    states[i] = new_state
                    trails[i].append((new_state.x, new_state.y))
                    times[i] += sim_dt

                    rho = math.hypot(new_state.x, new_state.y)
                    if rho < 0.18 and abs(math.atan2(math.sin(new_state.theta),
                                                     math.cos(new_state.theta))) < math.radians(8):
                        arrived[i] = True

        # ---------- draw ----------
        draw_grid(screen)
        draw_goal(screen)

        # trails
        for i, ctrl in enumerate(controllers):
            draw_trail(screen, trails[i], ctrl.color)

        # bikes
        for i, ctrl in enumerate(controllers):
            draw_bike(screen, states[i], params, ctrl.color)

        # ---------- HUD ----------
        # title
        title = title_font.render("Bicycle stabilization to (0, 0, 0)",
                                  True, FG)
        screen.blit(title, (24, 18))
        sub_t = small.render(
            "SPACE: random  •  R: reset  •  1/2/3: presets  "
            "•  TAB: pause  •  +/-: speed  •  Q: quit", True, SUB)
        screen.blit(sub_t, (24, 50))

        # start info
        start_t = small.render(
            f"start  x={start.x:+.2f}  y={start.y:+.2f}  "
            f"theta={math.degrees(start.theta):+.0f}°", True, SUB)
        screen.blit(start_t, (24, 70))

        # legend / metrics
        x0, y0 = 24, H - 30 - 22*len(controllers)
        for i, ctrl in enumerate(controllers):
            y = y0 + i * 22
            pygame.draw.rect(screen, ctrl.color, (x0, y + 4, 14, 14))
            ms = metrics[i]
            vmean = sum(ms["v"]) / max(1, len(ms["v"]))
            wmean = sum(ms["w"]) / max(1, len(ms["w"]))
            status = "DONE" if arrived[i] else f"t={times[i]:5.2f}s"
            line = (f"{ctrl.name:<18s}  {status:>10s}   "
                    f"len={ms['len']:5.2f} m   "
                    f"|v|={vmean:.2f}   |w|={wmean:.2f}")
            txt = font.render(line, True, FG)
            screen.blit(txt, (x0 + 22, y))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
