"""Grasp check: each item over several seeds and random orientations. Small items go in the tote, big ones
are carried, cans with drink stay upright in the hand.

    python scripts/grasp_check.py sock_1,can_1,sneaker_l,paper_bag_1 [seeds]
"""
import math
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homebody import baselines as B, items as I  # noqa: E402
from scripts import record_tasks as RT  # noqa: E402

items = sys.argv[1].split(",")
seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
for item in items:
    big = I.ITEMS[I.ITEM_NAMES.index(item)].shape == "hollow"
    res = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        env = RT.lab(seed + 1)
        RT.place(env, (2.0, -1.6), 0.0)
        k = RT.drop_item(env, item)
        qa = env.items.adr[k][0]
        yaw = rng.uniform(-math.pi, math.pi)
        env.data.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(env.model, env.data)
        B.run(env, 1.0, lambda e: B.command(e))
        body = env.items.body[k]
        t0 = time.time()
        if big:
            B.pick_and_carry(env, body)
            ok = B._lowest_z(env, body) > 0.1
        else:
            fill = env.items.fill[k]
            B.pick_into_tote(env, body)
            B.run(env, 1.0, lambda e: B.command(e, servos=B.servo_duty(e, env.holding, grip=1))
                  if getattr(e, "holding", None) is not None else B.command(e))
            if fill > 0.05:          # a drink: must be held upright, not tossed
                ok = env.items.spilled.sum() == 0 and env.data.xpos[body][2] > 0.3 and env.data.xmat[body][8] > 0.9
            else:
                ok = env.items.locate(env.data)[k] == "tote"
            print(f"    fill {fill:.2f} weighed {getattr(env, 'last_weight', float('nan')) * 1000:.0f} g", flush=True)
        res.append(ok)
        print(f"  {item} seed {seed} yaw {math.degrees(yaw):5.0f}: {'OK' if ok else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
    print(f"{item}: {sum(res)}/{len(res)}", flush=True)
