"""Drive the robot yourself in the MuJoCo viewer.

    python scripts/view.py                 # your scanned home, terrain level 1
    python scripts/view.py --scan          # ...drawn with the textured scan
    python scripts/view.py --terrain 0     # flat lab floor

Keys (click the viewer window first):
    arrows      drive / turn            Page Up / Page Down   lift up / down
    J / L       telescope out / in      U / O                 wrist yaw
    I / K       wrist pitch             N / M                 wrist roll
    G           toggle gripper          T / B                 head tilt up / down
    Y / H       head pan                Space                 stop the base
    R           reset (new home layout)
"""
import argparse
import math
import os
import sys
import time

import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homebody import EnvConfig, HomebodyEnv  # noqa: E402
from homebody import baselines as B  # noqa: E402
from homebody import spec as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--terrain", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--push-rate", type=float, default=0.0, help="accidental shoves per minute")
ap.add_argument("--scan", action="store_true", help="draw the textured scan of the home")
a = ap.parse_args()

env = HomebodyEnv(EnvConfig(terrain_level=a.terrain, push_rate=a.push_rate, reward="chores", scan_visual=a.scan))
obs, info = env.reset(seed=a.seed)
state = {"v": 0.0, "w": 0.0, "lift": 0.0, "arm": 0.0, "q": B.STOW_Q.copy(), "grip": False, "reset": False}
STEP = math.radians(6)


def key(k):
    s = state
    q = s["q"]
    if k == 265: s["v"] = min(s["v"] + 0.1, 0.6)
    elif k == 264: s["v"] = max(s["v"] - 0.1, -0.4)
    elif k == 263: s["w"] = min(s["w"] + 0.3, 1.5)
    elif k == 262: s["w"] = max(s["w"] - 0.3, -1.5)
    elif k == 32: s["v"] = s["w"] = 0.0
    elif k == 266: s["lift"] = 1.0 if s["lift"] <= 0 else 0.0
    elif k == 267: s["lift"] = -1.0 if s["lift"] >= 0 else 0.0
    elif k == ord("J"): s["arm"] = 1.0 if s["arm"] <= 0 else 0.0
    elif k == ord("L"): s["arm"] = -1.0 if s["arm"] >= 0 else 0.0
    elif k == ord("U"): q[0] += STEP
    elif k == ord("O"): q[0] -= STEP
    elif k == ord("I"): q[1] += STEP
    elif k == ord("K"): q[1] -= STEP
    elif k == ord("N"): q[2] += STEP
    elif k == ord("M"): q[2] -= STEP
    elif k == ord("G"): s["grip"] = not s["grip"]
    elif k == ord("T"): q[5] -= STEP
    elif k == ord("B"): q[5] += STEP
    elif k == ord("Y"): q[4] += STEP
    elif k == ord("H"): q[4] -= STEP
    elif k == ord("R"): s["reset"] = True


print(__doc__)
with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key) as v:
    for h in range(env.model.nhfield):
        v.update_hfield(h)
    last = 0.0
    while v.is_running():
        t = time.time()
        q = state["q"].copy()
        q[3] = 0.0 if state["grip"] else 0.055
        act = B.command(env, wheels=B.wheel_duty(env, state["v"], state["w"]), lift=state["lift"], arm=state["arm"],
                        servos=B.servo_duty(env, q, grip=1 if state["grip"] else None))
        obs, r, term, trunc, info = env.step(act)
        v.sync()
        if time.time() - last > 1.0:
            last = time.time()
            print(f"\r{info['room'] or '?':8s} battery {info['soc'] * 100:5.1f}%  {info['power_w']:5.1f} W  "
                  f"tidy {info['tidy']}/{info['items_active']}  "
                  f"{'CHARGING ' if info['charging'] else ''}{'DEAD' if info['dead'] else ''}", end="")
        if term or trunc or state["reset"]:
            obs, info = env.reset()
            for h in range(env.model.nhfield):
                v.update_hfield(h)
            state.update(v=0.0, w=0.0, lift=0.0, arm=0.0, q=B.STOW_Q.copy(), grip=False, reset=False)
        time.sleep(max(0.0, env.dt - (time.time() - t)))
