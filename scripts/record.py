"""Record scripted chores (sim-assisted) in the owner's recreated home.

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
from homebody import home as H  # noqa: E402
from homebody import items as I  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="media/demo.mp4")

a = ap.parse_args()

env = HomebodyEnv(EnvConfig(terrain_level=0.0, randomize_physics=False, sensor_noise=False, kinds=(),
                            ))
env.reset(seed=1)
for dd in env.doors.values():
    env.data.qpos[dd["q"]] = 0.0
r = mujoco.Renderer(env.model, 544, 960)
env._sync_hfield(r)
cams = {"pick": mujoco.MjvCamera(), "door": mujoco.MjvCamera()}
for c, (dist, el, az) in zip(cams.values(), ((1.8, -22, 135), (2.2, -18, 200))):
    c.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    c.trackbodyid = env.robot
    c.distance, c.elevation, c.azimuth = dist, el, az
view = {"cam": cams["pick"], "text": ""}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
w = imageio.get_writer(a.out, fps=20)
step = env.step


def recording_step(action):
    out = step(action)
    info = out[4]
    if not isinstance(view["cam"], str):
        view["cam"].lookat[:] = env.data.xpos[env.robot] + [0, 0, 0.5]
    r.update_scene(env.data, camera=view["cam"])
    img = Image.fromarray(r.render())
    dr = ImageDraw.Draw(img)
    dr.rectangle([12, 12, 560, 58], fill=(0, 0, 0))
    dr.text((20, 18), view["text"], fill=(255, 255, 255))
    dr.text((20, 36), f"{info['room']}   battery {info['soc'] * 100:4.1f}%   {info['power_w']:4.0f} W   "
                      f"laundry door {math.degrees(info['doors']['laundry_door']):3.0f} deg", fill=(200, 200, 200))
    w.append_data(np.asarray(img))
    return out


env.step = recording_step
view["text"] = "picking a can off the floor into the deck tote"
env._pose_robot(np.array(H.A(2.0, -1.6)), 0.0)
mujoco.mj_forward(env.model, env.data)
k = I.ITEM_NAMES.index("can_1")
qa, va = env.items.adr[k]
env.items.active[k] = True
R = env.data.xmat[env.robot].reshape(3, 3)
p = env.data.xpos[env.robot] + R @ np.array([-0.12, -0.45, 0])
env.data.qpos[qa:qa + 7] = [p[0], p[1], 0.07, 1, 0, 0, 0]
mujoco.mj_forward(env.model, env.data)
env.items.set_fill(k, 0.0)                      # an empty can
B.run(env, 1, lambda e: B.command(e))
B.pick_into_tote(env, env.items.body[k])
view["text"] = "laundry closet: gripping the knob, pulling the door off its catch"
view["cam"] = "laundry"                      # fixed camera aimed at the laundry closet
env._pose_robot(np.array(H.A(1.89, -3.44)), math.pi / 2)
mujoco.mj_forward(env.model, env.data)
B.pull_knob_door(env)
w.close()
print("wrote", a.out)
