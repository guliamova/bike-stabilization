"""Render the bicycle stabilization demo to a GIF (no display required)."""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import imageio.v2 as imageio

from bike import BikeState, BikeParams, step
from bike.controllers import ALL_CONTROLLERS
import main as M


def render_episode(start: BikeState, out_path: Path,
                   t_max: float = 12.0, fps: int = 30,
                   sim_dt: float = 0.02,
                   downscale: int = 2) -> None:
    pygame.init()
    screen = pygame.display.set_mode((M.W, M.H))
    title_font = pygame.font.SysFont("georgia", 22, bold=True)
    font  = pygame.font.SysFont("consolas", 16)
    small = pygame.font.SysFont("consolas", 13)

    params = BikeParams()
    controllers = [C(params) for C in ALL_CONTROLLERS]
    states = [BikeState(start.x, start.y, start.theta) for _ in controllers]
    trails = [[(s.x, s.y)] for s in states]
    times = [0.0 for _ in controllers]
    arrived = [False for _ in controllers]
    metrics = [{"len": 0.0, "v": [], "w": []} for _ in controllers]

    # Sim runs at 1/sim_dt steps per second; we save every (1/(fps*sim_dt))-th
    steps_per_frame = max(1, int(round(1.0 / (fps * sim_dt))))
    total_frames = int(t_max * fps)

    frames = []
    sim_t = 0.0

    for f in range(total_frames):
        # advance sim
        for _ in range(steps_per_frame):
            if sim_t >= t_max:
                break
            for i, ctrl in enumerate(controllers):
                if arrived[i]:
                    continue
                v, d = ctrl.control(states[i])
                ns = step(states[i], v, d, sim_dt, params)
                metrics[i]["len"] += math.hypot(ns.x-states[i].x, ns.y-states[i].y)
                metrics[i]["v"].append(abs(v))
                metrics[i]["w"].append(abs(v/params.wheelbase*math.tan(d)))
                states[i] = ns
                trails[i].append((ns.x, ns.y))
                times[i] += sim_dt
                rho = math.hypot(ns.x, ns.y)
                if rho < 0.18 and abs(math.atan2(math.sin(ns.theta),
                                                  math.cos(ns.theta))) < math.radians(8):
                    arrived[i] = True
            sim_t += sim_dt

        # draw
        M.draw_grid(screen)
        M.draw_goal(screen)
        for i, ctrl in enumerate(controllers):
            M.draw_trail(screen, trails[i], ctrl.color)
        for i, ctrl in enumerate(controllers):
            M.draw_bike(screen, states[i], params, ctrl.color)

        # HUD
        screen.blit(title_font.render("Bicycle stabilization to (0, 0, 0)",
                                       True, M.FG), (24, 18))
        screen.blit(small.render(
            f"start  x={start.x:+.2f}  y={start.y:+.2f}  "
            f"theta={math.degrees(start.theta):+.0f}°", True, M.SUB), (24, 50))

        x0, y0 = 24, M.H - 30 - 22*len(controllers)
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
            screen.blit(font.render(line, True, M.FG), (x0 + 22, y))

        # capture frame
        arr = pygame.surfarray.array3d(screen).swapaxes(0, 1)  # (H, W, 3)
        if downscale > 1:
            from PIL import Image
            img = Image.fromarray(arr).resize(
                (M.W // downscale, M.H // downscale), Image.LANCZOS)
            arr = __import__("numpy").array(img)
        frames.append(arr)

    pygame.quit()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    # Use first frame to build a high-quality adaptive palette, then map
    # all subsequent frames to it -> stable colors across frames, no
    # palette flicker.
    rgb_frames = [Image.fromarray(arr).convert("RGB") for arr in frames]
    palette = rgb_frames[0].quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pil_frames = [
        f.quantize(palette=palette, dither=Image.Dither.NONE)
        for f in rgb_frames
    ]
    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        duration=int(1000/fps), loop=0, optimize=False, disposal=2,
    )
    print(f"saved -> {out_path}  ({len(frames)} frames, "
          f"{M.W//downscale}x{M.H//downscale})")


if __name__ == "__main__":
    render_episode(
        BikeState(-4.0, 2.5, math.radians(-30)),
        Path("results/demo.gif"),
        t_max=9.0, fps=15, downscale=2,
    )
