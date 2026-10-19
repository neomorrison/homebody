"""Watch the robot live in the MuJoCo viewer.

    python scripts/view.py --brain hold      # posture reflex
    python scripts/view.py --brain babble    # random muscle twitches
    python scripts/view.py --brain limp
"""
import argparse
import os
import sys
import time

import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homebody import EnvConfig, HomebodyEnv  # noqa: E402
from homebody import baselines as B  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--brain", default="hold", choices=["hold", "babble", "limp"])
ap.add_argument("--start", default="standing", choices=["standing", "lying", "random"])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

env = HomebodyEnv(EnvConfig(start=a.start))
obs, info = env.reset(seed=a.seed)
brain = {"hold": B.make_hold(), "babble": B.make_random(), "limp": B.limp}[a.brain]
with mujoco.viewer.launch_passive(env.model, env.data) as v:
    while v.is_running():
        t = time.time()
        obs, r, term, trunc, info = env.step(brain(obs))
        v.sync()
        if term or trunc:
            print("died" if term else "time up", {k: info[k] for k in ("t", "soc", "damage")})
            obs, info = env.reset()
        time.sleep(max(0.0, env.dt - (time.time() - t)))
