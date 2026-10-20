"""Physical checks on the robot in the owner's recreated home. Run: python scripts/smoke_test.py"""
import math
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homebody import EnvConfig, HomebodyEnv  # noqa: E402
from homebody import baselines as B  # noqa: E402
from homebody import home as H  # noqa: E402
from homebody import items as I  # noqa: E402

results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def lab(seed=0, **kw):
    """The real home with a flat floor, no clutter, doors shut, no noise."""
    base = dict(terrain_level=0, randomize_physics=False, sensor_noise=False, kinds=())
    base.update(kw)
    env = HomebodyEnv(EnvConfig(**base))
    env.reset(seed=seed)
    for dd in env.doors.values():
        env.data.qpos[dd["q"]] = 0.0
    for q, _ in env.mdoor.values():
        env.data.qpos[q] = 0.0
    return env


def place(env, aligned_xy, yaw):
    env._pose_robot(np.array(H.A(*aligned_xy), float), yaw)
    mujoco.mj_forward(env.model, env.data)


t0 = time.time()

env = lab()
place(env, (-2.2, 0.5), 0.0)                     # in the hallway, facing the living room
p0 = env.data.qpos[env.rq:env.rq + 2].copy()
info = B.run(env, 3, lambda e: B.command(e, wheels=B.wheel_duty(e, 0.4, 0)))
dist = np.linalg.norm(env.data.qpos[env.rq:env.rq + 2] - p0)
check("drives down the hallway", dist > 0.9 and info["damage" if "damage" in info else "upright"] != 0,
      f"{dist:.2f} m in 3 s at 0.4 m/s, {info['power_w']:.0f} W, room {info['room']}")

env = lab()
place(env, (2.0, -1.6), 0.0)                     # foyer
info = B.run(env, 13, lambda e: B.command(e, lift=B.lift_cmd(e, 1.45), arm=B.arm_cmd(e, 0.5)))
h = info["lift_height"]
env.battery.dead = True
info = B.run(env, 3, lambda e: np.zeros(10))
env.battery.dead = False
check("lift reaches dryer height + counterbalance", h > 1.5 and abs(info["lift_height"] - h) < 0.02,
      f"telescope up to {h:.2f} m (dryer door centre 1.55 m), after a 3 s power cut {info['lift_height']:.2f} m")

env = lab(1)
place(env, (2.0, -1.6), 0.0)
k = I.ITEM_NAMES.index("can_1")
qa, va = env.items.adr[k]
env.items.active[k] = True
R = env.data.xmat[env.robot].reshape(3, 3)
w = R @ np.array([-0.12, -0.45, 0])
env.data.qpos[qa:qa + 7] = [env.data.xpos[env.robot][0] + w[0], env.data.xpos[env.robot][1] + w[1], 0.07, 1, 0, 0, 0]
mujoco.mj_forward(env.model, env.data)
B.run(env, 1, lambda e: B.command(e))
B.pick_into_tote(env, env.items.body[k])
check("picks a can off the floor into its tote", env.items.locate(env.data)[k] == "tote",
      f"can ended in: {env.items.locate(env.data)[k]}")

env = lab()
place(env, (1.89, -3.44), math.pi / 2)          # foyer, beside the laundry closet, facing north
info = B.pull_knob_door(env)
ang = math.degrees(info["doors"]["laundry_door"])
check("pulls the laundry closet door off its catch", ang > 8, f"door open {ang:.0f} deg after a 20 cm pull")

env = lab(soc_range=(0.4, 0.4))
env.put_on_dock()
info = B.run(env, 5, lambda e: B.command(e))
check("charges on the foyer dock", info["charging"] and info["soc"] > 0.4,
      f"{info['charge_a']:.2f} A in, {info['load_a']:.2f} A out")

env = lab(soc_range=(0.3, 0.3), capacity_scale=0.0015)
info = B.run(env, 120, lambda e: B.command(e))
check("an empty battery is death", info["battery_dead"] and info["dead"], f"died at t={info['t']:.0f} s")

for force in (80.0, 200.0):
    tipped, worst = 0, 1.0
    for seed in range(8):
        env = lab(seed)
        place(env, (2.0, -1.2), 0.0)
        B.run(env, 10, lambda e: B.command(e, lift=B.lift_cmd(e, 1.4), arm=B.arm_cmd(e, 0.5)))
        a = 2 * math.pi * seed / 8
        env.push = (np.array([force * math.cos(a), force * math.sin(a), 0.0]), env.data.xpos[env.head_body].copy())
        env.push_left = 0.2
        info = B.run(env, 3, lambda e: B.command(e))
        tipped += info["tipped"]
        worst = min(worst, info["upright"])
    lean = math.degrees(math.acos(max(min(worst, 1), -1)))
    if force < 100:
        check(f"survives an {force:.0f} N shove at the mast top (lift up, arm out)", tipped == 0,
              f"tipped {tipped}/8, worst lean {lean:.1f} deg")
    else:
        print(f"       limit: a {force:.0f} N shove tips it {tipped}/8 times")

env = HomebodyEnv(EnvConfig())
_, info = env.reset(seed=0)
placed = info["items_active"]
B.run(env, 2, lambda e: B.command(e))
lost = [n for n, k in zip(I.ITEM_NAMES, range(len(I.ITEMS))) if env.items.active[k] and env.data.xpos[env.items.body[k]][2] < -0.05]
check("the owner's clutter is in the house", placed >= len(I.ITEMS) - 2 and not lost,
      f"{placed}/{len(I.ITEMS)} items placed near where they were scanned (crowded ones are left out); "
      f"fell through: {lost or 'none'}")
t = time.time()
for _ in range(100):
    env.step(np.zeros(10))
sps = 100 / (time.time() - t)
print(f"       sim speed with all clutter out: {sps:.0f} control steps/s = {sps / 20:.1f}x real time (one CPU core)")
print(f"{sum(results)}/{len(results)} passed in {time.time() - t0:.0f} s")
sys.exit(0 if all(results) else 1)
