"""Checks that the body and world behave like the physical thing should.

    python scripts/smoke_test.py
"""
import math
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homebody import EnvConfig, HomebodyEnv  # noqa: E402
from homebody import baselines as B  # noqa: E402


def run(env, brain, seconds):
    obs, info = env.obs0, env.info
    for _ in range(int(seconds * 50)):
        obs, r, term, trunc, info = env.step(brain(obs))
        if term:
            break
    return info


def fresh(**kw):
    env = HomebodyEnv(EnvConfig(randomize_physics=False, **kw))
    env.obs0, env.info = env.reset(seed=kw.pop("seed", 0) if "seed" in kw else 0)
    return env


def put_on_dock(env):
    m, d = env.model, env.data
    q = m.body_quat[env.dock]
    yaw = 2 * math.atan2(q[3], q[0])
    tgt = d.site_xpos[env.dock_target]
    d.qpos[0:2] = tgt[:2] - 0.02 * np.array([math.cos(yaw), math.sin(yaw)])
    d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    d.qpos[2] += 0.01
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    env.obs0 = env._obs()


results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


t0 = time.time()
env = fresh()
info = run(env, B.limp, 3)
check("limp body collapses", info["fallen"], f"height {info['height']:.3f} m")

env = fresh()
info = run(env, B.make_hold(), 10)
w = info["load_a"] * info["v"]
check("posture reflex holds it up", not info["fallen"],
      f"height {info['height']:.3f} m, draw {w:.1f} W, hottest servo {info['servo_temp_max']:.1f} C")
hours = 3.5 * 11.1 / max(w, 1e-6)
print(f"       -> a full pack would hold this crouch for about {hours:.1f} h")

env = fresh(soc_range=(0.4, 0.4))
put_on_dock(env)
info = run(env, B.make_hold(), 10)
check("standing on the dock charges", info["charging"] and info["soc"] > 0.4,
      f"charge {info['charge_a']:.2f} A in, {info['load_a']:.2f} A out, soc 0.400 -> {info['soc']:.4f}")
ir = env._ir()
check("dock beacon visible from the dock", ir.max() > 0, f"IR receivers {np.round(ir, 3)}")

env = fresh(soc_range=(0.4, 0.4), capacity_scale=0.002)
info = run(env, B.make_hold(), 120)
check("empty battery kills it", info["battery_dead"] and info["dead"],
      f"died at t={info['t']:.1f} s (pack scaled to 0.2%), V={info['v']:.2f}")

env = fresh(start="lying")
info = run(env, B.make_random(np.random.default_rng(0)), 5)
check("motor babbling runs from lying down", True,
      f"upright {info['upright']:.2f}, damage {info['damage']:.2f} N s, used {info['energy_used_wh'] * 1000:.1f} mWh")

# --- real-floor checks -------------------------------------------------------
env = HomebodyEnv(EnvConfig(terrain_level=1.0))
mus, rms, tilts, kinds = [], [], [], set()
for seed in range(12):
    _, info0 = env.reset(seed=seed)
    ts = info0["terrain"]
    rms.append(ts.bump_rms_mm)
    tilts.append(ts.tilt_deg)
    kinds.add(ts.floor)
    run_env = env
    o = env._obs()
    for _ in range(25):
        o, *_ = env.step(B.make_hold()(o))
    g = env.data.contact.geom
    for i, (a, b) in enumerate(g):
        if {a, b} & set(env.sole.values()) and env.terrain.floor_geom in (a, b):
            mus.append(env.data.contact.friction[i][0])
            break
check("floor friction is what the soles feel", len(mus) > 0 and max(mus) - min(mus) > 0.2,
      f"sole-floor mu over {len(mus)} homes: {min(mus):.2f}..{max(mus):.2f}; floors seen {sorted(kinds)}")
check("level-1 floor is uneven like a house", 1.0 < np.mean(rms) < 6.0,
      f"bumps {np.mean(rms):.1f} mm RMS, tilt up to {max(tilts):.2f} deg")

env = HomebodyEnv(EnvConfig(terrain_level=0.0))
_, info0 = env.reset(seed=0)
check("level-0 floor is flat", info0["terrain"].bump_rms_mm < 1e-6, f"RMS {info0['terrain'].bump_rms_mm:.3f} mm, {info0['terrain'].floor}")

env = fresh(start="lying")
env.data.qpos[2] += 0.8          # dropped from about 90 cm
mujoco.mj_forward(env.model, env.data)
env.obs0 = env._obs()
info = run(env, B.limp, 2)
check("a drop from 90 cm does damage", info["damage"] > 0, f"damage {info['damage']:.1f} N s (limit {env.cfg.damage_limit})")

env = fresh()
t = time.time()
o = env.obs0
for _ in range(500):
    o, *_ = env.step(np.zeros(20))
sps = 500 / (time.time() - t)
print(f"       sim speed {sps:.0f} control steps/s = {sps / 50:.0f}x real time (one CPU core)")
print(f"{sum(results)}/{len(results)} passed in {time.time() - t0:.1f} s")
sys.exit(0 if all(results) else 1)
