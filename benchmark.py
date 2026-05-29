"""Run all controllers from N random start poses and dump stats."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from bike import BikeState, BikeParams
from bike.controllers import ALL_CONTROLLERS
from bike.simulator import run


def random_start(rng: random.Random) -> BikeState:
    x = rng.uniform(-4.5, 4.5)
    y = rng.uniform(-4.5, 4.5)
    th = rng.uniform(-math.pi, math.pi)
    return BikeState(x, y, th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="random starts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t-max", type=float, default=30.0)
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    starts = [random_start(rng) for _ in range(args.n)]
    params = BikeParams()

    results = {}
    for C in ALL_CONTROLLERS:
        rows = []
        n_ok = 0
        for s in starts:
            ctrl = C(params)
            ep = run(ctrl, s, params, t_max=args.t_max, pos_tol=0.3)
            rows.append({
                "start": [s.x, s.y, s.theta],
                "success": ep.success,
                "time_to_goal": ep.time_to_goal,
                "path_length": ep.path_length,
                "mean_lin_speed": ep.mean_lin_speed,
                "mean_ang_speed": ep.mean_ang_speed,
                "final_pos_err": ep.final_pos_err,
                "final_head_err": ep.final_head_err,
            })
            n_ok += int(ep.success)
        results[C.__name__] = {
            "name": C.name,
            "color": list(C.color),
            "success_rate": n_ok / len(starts),
            "rows": rows,
        }
        print(f"{C.__name__:18s}: {n_ok}/{len(starts)}  "
              f"({100*n_ok/len(starts):.0f}% reached position)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
