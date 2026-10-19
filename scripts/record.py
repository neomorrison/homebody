"""Record the scripted chores (sim-assisted) to a video.

    python scripts/record.py --out media/demo.mp4
"""
import argparse
import math
import os
import sys

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homebody import EnvConfig, HomebodyEnv  # noqa: E402
from homebody import baselines as B  # noqa: E402
from homebody import items as I  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="media/demo.mp4")
a = ap.parse_args()

env = HomebodyEnv(EnvConfig(terrain_level=0.0, randomize_physics=False, sensor_noise=False, n_trash=0, n_clothes=0, n_pods=4,
                            door_open_prob=0))
env.reset(seed=1)
r = mujoco.Renderer(env.model, 544, 960)
env._sync_hfield(r)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = env.robot
cam.distance, cam.elevation, cam.azimuth = 1.7, -20, 60
view = {"camera": "demo_pick"}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
w = imageio.get_writer(a.out, fps=20)
caption = {"text": ""}
step = env.step


def recording_step(action):
    out = step(action)
    info = out[4]
    cam.lookat[:] = env.data.xpos[env.robot] + [0, 0, 0.45]
    r.update_scene(env.data, camera=view["camera"])
    img = Image.fromarray(r.render())
    dr = ImageDraw.Draw(img)
    dr.rectangle([12, 12, 520, 58], fill=(0, 0, 0))
    dr.text((20, 18), caption["text"], fill=(255, 255, 255))
    dr.text((20, 36), f"battery {info['soc'] * 100:4.1f}%  {info['power_w']:4.0f} W  door {math.degrees(info['door_angle']):3.0f} deg",
            fill=(200, 200, 200))
    w.append_data(np.asarray(img))
    return out


env.step = recording_step
caption["text"] = "picking a paper ball off the floor into the deck tote"
env._pose_robot(np.array([-0.5, -0.3]), 0.0)
mujoco.mj_forward(env.model, env.data)
k = I.ITEM_NAMES.index("paper_ball_1")
qa, va = env.items.adr[k]
env.items.active[k] = True
R = env.data.xmat[env.robot].reshape(3, 3)
p = env.data.xpos[env.robot] + R @ np.array([-0.12, -0.45, 0])
env.data.qpos[qa:qa + 7] = [p[0], p[1], 0.06, 1, 0, 0, 0]
mujoco.mj_forward(env.model, env.data)
B.run(env, 1, lambda e: B.command(e))
B.pick_into_tote(env, env.items.body[k])
caption["text"] = "opening the laundry door: press the lever, push"
view["camera"] = "door"
env._pose_robot(np.array([3.86, -0.78]), 0.0)
mujoco.mj_forward(env.model, env.data)
B.open_lever_door(env)
w.close()
print("wrote", a.out)
