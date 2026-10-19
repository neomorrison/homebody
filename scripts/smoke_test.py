"""Physical checks on the robot and the home. Run: python scripts/smoke_test.py"""
import math
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homebody import EnvConfig, HomebodyEnv  # noqa: E402
from homebody import baselines as B  # noqa: E402
from homebody import items as I  # noqa: E402

results = []


def check(name, ok, detail):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def lab(seed=0, **kw):
    base = dict(terrain_level=0, randomize_physics=False, sensor_noise=False, n_trash=0, n_clothes=0, n_pods=0,
                door_open_prob=0)
    base.update(kw)
    env = HomebodyEnv(EnvConfig(**base))
    env.reset(seed=seed)
    return env


def place(env, xy, yaw):
    env._pose_robot(np.array(xy, float), yaw)
    mujoco.mj_forward(env.model, env.data)


t0 = time.time()

env = lab()
place(env, (-0.5, -0.3), 0.0)
p0 = env.data.qpos[env.rq:env.rq + 2].copy()
info = B.run(env, 3, lambda e: B.command(e, wheels=B.wheel_duty(e, 0.4, 0)))
dist = np.linalg.norm(env.data.qpos[env.rq:env.rq + 2] - p0)
check("drives", dist > 0.9, f"{dist:.2f} m in 3 s at 0.4 m/s, {info['power_w']:.0f} W")

info = B.run(env, 10, lambda e: B.command(e, lift=B.lift_cmd(e, 1.25), arm=B.arm_cmd(e, 0.5)))
h = info["lift_height"]
env.battery.dead = True
info = B.run(env, 3, lambda e: np.zeros(10))
env.battery.dead = False
check("lift + counterbalance", h > 1.3 and abs(info["lift_height"] - h) < 0.02,
      f"telescope up to {h:.2f} m, after a 3 s power cut {info['lift_height']:.2f} m (spring holds it)")

env = lab(1)
place(env, (-0.5, -0.3), 0.0)
k = I.ITEM_NAMES.index("paper_ball_1")
qa, va = env.items.adr[k]
env.items.active[k] = True
R = env.data.xmat[env.robot].reshape(3, 3)
w = R @ np.array([-0.12, -0.45, 0])
env.data.qpos[qa:qa + 7] = [env.data.xpos[env.robot][0] + w[0], env.data.xpos[env.robot][1] + w[1], 0.05, 1, 0, 0, 0]
mujoco.mj_forward(env.model, env.data)
B.run(env, 1, lambda e: B.command(e))
info = B.pick_into_tote(env, env.items.body[k])
check("picks trash off the floor into its tote", env.items.locate(env.data)[k] == "tote",
      f"paper ball ended in: {env.items.locate(env.data)[k]}")

env = lab()
place(env, (3.86, -0.78), 0.0)
locked = env.info["door_latched"]
info = B.open_lever_door(env)
check("opens the laundry door (lever, then push)", locked and info["door_angle"] > math.radians(10),
      f"latched at start: {locked}; door open {math.degrees(info['door_angle']):.0f} deg; wrist damage {info['part_damage']['wrist']}")

env = lab(soc_range=(0.4, 0.4))
env.put_on_dock()
info = B.run(env, 5, lambda e: B.command(e))
check("charges on the dock", info["charging"] and info["soc"] > 0.4,
      f"{info['charge_a']:.2f} A in, {info['load_a']:.2f} A out")

env = lab(soc_range=(0.3, 0.3), capacity_scale=0.0015)
info = B.run(env, 120, lambda e: B.command(e))
check("an empty battery is death", info["battery_dead"] and info["dead"], f"died at t={info['t']:.0f} s, {info['v']:.1f} V")

for force in (80.0, 200.0):
    tipped = 0
    worst = 1.0
    for seed in range(10):
        env = lab(seed, push_rate=0.0)
        place(env, (0.5, 0.0), 0.0)
        B.run(env, 8, lambda e: B.command(e, lift=B.lift_cmd(e, 1.2), arm=B.arm_cmd(e, 0.5)))
        a = env.np_random.uniform(-math.pi, math.pi)
        env.push = (np.array([force * math.cos(a), force * math.sin(a), 0.0]), env.data.xpos[env.head_body].copy())
        env.push_left = 0.2
        info = B.run(env, 3, lambda e: B.command(e))
        tipped += info["tipped"]
        worst = min(worst, info["upright"])
    lean = math.degrees(math.acos(max(min(worst, 1), -1)))
    if force < 100:
        check(f"survives an {force:.0f} N shove at the top of the mast (mast up, arm out)", tipped == 0,
              f"tipped {tipped}/10, worst lean {lean:.1f} deg")
    else:
        print(f"       limit: a {force:.0f} N shove tips it {tipped}/10 times (any tall narrow robot falls to a hard push)")

env = HomebodyEnv(EnvConfig(terrain_level=1.0))
rms, floors = [], set()
for seed in range(6):
    _, i0 = env.reset(seed=seed)
    rms.append(i0["terrain"].bump_rms_mm)
    floors.add(i0["terrain"].floor)
check("home floor is uneven and varied", 1.0 < np.mean(rms) < 6.0, f"bumps {np.mean(rms):.1f} mm RMS; floors {sorted(floors)}")

env = HomebodyEnv(EnvConfig())
env.reset(seed=0)
t = time.time()
for _ in range(200):
    env.step(np.zeros(10))
sps = 200 / (time.time() - t)
print(f"       sim speed {sps:.0f} control steps/s = {sps / 20:.1f}x real time (one CPU core)")
print(f"{sum(results)}/{len(results)} passed in {time.time() - t0:.0f} s")
sys.exit(0 if all(results) else 1)
