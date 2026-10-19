"""Render a video of a brain driving the robot, with a battery readout.

    python scripts/record.py --brain babble --seconds 10 --out media/babble.mp4
"""
import argparse
import os
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homebody import EnvConfig, HomebodyEnv  # noqa: E402
from homebody import baselines as B  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--brain", default="hold", choices=["hold", "babble", "limp"])
ap.add_argument("--start", default="standing")
ap.add_argument("--seconds", type=float, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--camera", default="track", help="track | overview | eye")
ap.add_argument("--out", default="media/out.mp4")
a = ap.parse_args()

import mujoco  # noqa: E402

env = HomebodyEnv(EnvConfig(start=a.start))
obs, info = env.reset(seed=a.seed)
brain = {"hold": B.make_hold(), "babble": B.make_random(np.random.default_rng(a.seed)), "limp": B.limp}[a.brain]
r = mujoco.Renderer(env.model, 544, 960)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = env.robot
cam.distance, cam.elevation, cam.azimuth = 1.4, -15, 135
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
w = imageio.get_writer(a.out, fps=25)
for i in range(int(a.seconds * 50)):
    obs, rew, term, trunc, info = env.step(brain(obs))
    if i % 2:
        continue
    if a.camera == "track":
        r.update_scene(env.data, camera=cam)
    else:
        r.update_scene(env.data, camera=a.camera)
    img = Image.fromarray(r.render())
    dr = ImageDraw.Draw(img)
    dr.rectangle([12, 12, 330, 74], fill=(0, 0, 0))
    dr.text((20, 18), f"t {info['t']:5.1f}s  battery {info['soc'] * 100:5.1f}%  {info['v']:.2f} V", fill=(255, 255, 255))
    dr.text((20, 36), f"draw {info['load_a'] * info['v']:5.1f} W   charging {'yes' if info['charging'] else 'no'}", fill=(255, 255, 255))
    dr.text((20, 54), f"dock {info['dock_dist']:.2f} m  damage {info['damage']:.1f}", fill=(255, 255, 255))
    w.append_data(np.asarray(img))
    if term:
        break
w.close()
print("wrote", a.out)
