"""Record the robot attempting chores in the owner's scanned home, for the GitHub Pages site.

    python scripts/record_tasks.py              # all tasks -> docs/videos/*.mp4 + docs/tasks.json
    python scripts/record_tasks.py dock shove   # just some

Every attempt is scripted (targets come from the simulator, not the cameras); the
outcome is measured, not assumed, and failures are kept.
"""
import json
import math
import os
import sys
import time

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from homebody import EnvConfig, HomebodyEnv  # noqa: E402
from homebody import baselines as B  # noqa: E402
from homebody import home as H  # noqa: E402
from homebody import items as I  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "videos")
W, HGT, FPS = 960, 544, 20


class Recorder:
    """Wraps env.step so every control step becomes a video frame with a caption."""

    def __init__(self, env, path):
        self.env, self.path = env, path
        self.r = mujoco.Renderer(env.model, HGT, W)
        env._sync_hfield(self.r)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = env.robot
        self.camera = self.cam
        self.caption = ""
        self.w = imageio.get_writer(path, fps=FPS, quality=7, macro_block_size=16)
        self._step = env.step
        env.step = self.step

    def follow(self, distance, elevation, azimuth, height=0.5):
        self.cam.distance, self.cam.elevation, self.cam.azimuth = distance, elevation, azimuth
        self.cam_height = height
        self.camera = self.cam

    def fixed(self, name):
        self.camera = name

    def step(self, action):
        out = self._step(action)
        info = out[4]
        if not isinstance(self.camera, str):
            self.cam.lookat[:] = self.env.data.xpos[self.env.robot] + [0, 0, getattr(self, "cam_height", 0.5)]
        self.r.update_scene(self.env.data, camera=self.camera)
        img = Image.fromarray(self.r.render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, HGT - 46, W, HGT], fill=(0, 0, 0))
        dr.text((14, HGT - 40), self.caption, fill=(255, 255, 255))
        dr.text((14, HGT - 22), f"t {info['t']:5.1f} s   {info['room'] or ''}   battery {info['soc'] * 100:4.1f}%   "
                                f"{info['power_w']:4.0f} W   {'CHARGING' if info['charging'] else ''}"
                                f"{'  TIPPED' if info['tipped'] else ''}", fill=(190, 190, 190))
        self.w.append_data(np.asarray(img))
        return out

    def hold(self, seconds):
        B.run(self.env, seconds, lambda e: B.command(e))

    def close(self):
        self.w.close()
        self.r.close()


def lab(seed=1, **kw):
    cfg = dict(terrain_level=0.5, randomize_physics=False, sensor_noise=False, kinds=(),
               soc_range=(0.6, 0.6))
    cfg.update(kw)
    env = HomebodyEnv(EnvConfig(**cfg))
    env.reset(seed=seed)
    for dd in env.doors.values():
        env.data.qpos[dd["q"]] = 0.0
    for q, _ in env.mdoor.values():
        env.data.qpos[q] = 0.0
    return env


def place(env, aligned_xy, yaw):
    env._pose_robot(np.array(H.A(*aligned_xy), float), yaw)
    mujoco.mj_forward(env.model, env.data)


def drop_item(env, name, local=(-0.12, -0.45)):
    k = I.ITEM_NAMES.index(name)
    qa, va = env.items.adr[k]
    env.items.active[k] = True
    R = env.data.xmat[env.robot].reshape(3, 3)
    p = env.data.xpos[env.robot] + R @ np.array([*local, 0])
    it = I.ITEMS[k]
    z = (it.size[1] if it.geom == "cylinder" else it.size[2]) + 0.02
    env.data.qpos[qa:qa + 7] = [p[0], p[1], z, 1, 0, 0, 0]
    env.data.qvel[va:va + 6] = 0
    mujoco.mj_forward(env.model, env.data)
    return k


# ----------------------------------------------------------------------------
# tasks: each returns (success: bool, what happened: str)
# ----------------------------------------------------------------------------

def task_pick(item, label, fill=0.0):
    def run(rec):
        env = rec.env
        place(env, (2.0, -1.6), 0.0)
        k = drop_item(env, item)
        if I.ITEMS[k].liquid:
            env.items.set_fill(k, fill)
        rec.follow(1.7, -24, 120)
        rec.caption = f"Pick the {label} off the floor and drop it in the deck tote"
        rec.hold(1.0)
        B.pick_into_tote(env, env.items.body[k])
        rec.hold(1.5)
        where = env.items.locate(env.data)[k]
        lifted = where == "tote"
        return lifted, (f"The {label} ended up in the tote." if lifted else
                        f"The {label} was not picked up (it ended up {where or 'on the floor'}).")
    return run


def task_full_can(rec):
    env = rec.env
    place(env, (2.0, -1.6), 0.0)
    k = drop_item(env, "can_1")
    env.items.set_fill(k, 0.5)
    rec.follow(1.7, -24, 120)
    rec.caption = "A can that is still half full: pick it up, weigh it on the wrist, decide"
    rec.hold(1.0)
    B.pick_into_tote(env, env.items.body[k])
    kg = env.last_weight
    if getattr(env, "holding", None) is None:
        return False, f"Weighed {kg * 1000:.0f} g but treated it as empty and dropped it in the tote."
    rec.caption = f"Weighs {kg * 1000:.0f} g: there's drink in it. Keep it upright and carry it to the kitchen"
    rec.follow(2.2, -50, 120)
    B.carry(env, env.holding, v=0.25, seconds=4.0)
    rec.hold(1.0)
    up = env.data.xmat[env.items.body[k]][8]
    ok = env.info["spills"] == 0 and env.data.xpos[env.items.body[k]][2] > 0.3 and up > 0.9
    return ok, (f"Weighed {kg * 1000:.0f} g (an empty can is 15 g), so it kept the can upright and carried it "
                f"off without spilling." if ok else "The drink spilled.")


def task_carry_bag(rec):
    env = rec.env
    place(env, (2.0, -1.6), 0.0)
    k = drop_item(env, "paper_bag_1")
    rec.follow(1.9, -20, 120)
    rec.caption = "A paper bag is too big for the tote: pinch the rim and carry it to recycling"
    rec.hold(1.0)
    body = env.items.body[k]
    B.pick_and_carry(env, body, v=0.25, seconds=4.0)
    rec.hold(1.0)
    held = B._lowest_z(env, body) > 0.1
    return held, ("Pinched the rim, lifted the bag and carried it 1 m without dropping it." if held else
                  "The bag slipped out of the pinch.")


def task_drive(rec):
    env = rec.env
    env.data.qpos[env.doors["bedroom_door"]["q"]] = math.radians(88)
    place(env, (-3.55, 0.5), 0.0)
    rec.fixed("hallway")
    rec.caption = "Drive out of the bedroom and down the 1 m hallway (sill in the doorway)"
    start = env.data.qpos[env.rq:env.rq + 2].copy()
    target_y = H.A(0, 0.5)[1]
    info = env.info
    for _ in range(int(14 * 20)):
        y, yaw = env.data.qpos[env.rq + 1], env.info["yaw"]
        w = np.clip(-2.0 * (y - target_y) - 2.5 * yaw, -0.6, 0.6)            # stay on the hallway centreline
        _, _, term, _, info = env.step(B.command(env, wheels=B.wheel_duty(env, 0.35, w)))
        if term or env.data.qpos[env.rq] > H.A(0.1, 0)[0]:
            break
    dist = float(np.linalg.norm(env.data.qpos[env.rq:env.rq + 2] - start))
    ok = dist > 3.0 and not info["dead"]
    return ok, f"Drove {dist:.1f} m from the bedroom to the end of the hallway, {info['power_w']:.0f} W."


def task_dock(rec):
    env = rec.env
    m, d = env.model, env.data
    mujoco.mj_kinematics(m, d)
    tgt = d.site_xpos[env.dock_target][:2].copy()
    (_, _), yaw = H.DOCK_POSE
    start = tgt + 1.2 * np.array([math.cos(yaw), math.sin(yaw)]) + [0.0, 0.12]
    env._pose_robot(start, yaw + 0.15)
    mujoco.mj_forward(m, d)
    rec.follow(2.6, -58, 200, 0.3)
    rec.caption = "Back onto the charging dock (one pad on each spring contact)"
    info = env.info
    for _ in range(int(16 * 20)):
        p = d.qpos[env.rq:env.rq + 2]
        heading = np.array([math.cos(yaw), math.sin(yaw)])
        lateral = float(np.cross(heading, p - tgt))
        dyaw = (env.info["yaw"] - yaw + math.pi) % (2 * math.pi) - math.pi
        along = float(heading @ (p - tgt))
        v = -0.15 if along > 0.12 else -0.06
        w = np.clip(-3.0 * dyaw + 4.0 * lateral, -0.5, 0.5)                   # reversing: steer the tail in
        _, _, term, _, info = env.step(B.command(env, wheels=B.wheel_duty(env, v, w)))
        if info["charging"]:
            break
    rec.caption = "Docked: charging" if info["charging"] else "Did not reach the contacts"
    rec.hold(2.0)
    info = env.info
    return bool(info["charging"]), (f"Charging at {info['charge_a']:.1f} A after backing in from 1.2 m."
                                    if info["charging"] else "Missed the contacts.")


def task_door(rec):
    env = rec.env
    place(env, (1.89, -3.44), math.pi / 2)
    rec.fixed("laundry")
    rec.caption = "Laundry closet: grip the knob and pull the door off its magnetic catch"
    rec.hold(0.5)
    info = B.pull_knob_door(env)
    rec.hold(1.0)
    ang = math.degrees(env.info["doors"]["laundry_door"])
    return ang > 8, f"Door opened {ang:.0f} degrees (the robot's own body blocks more in this layout)."


def task_dryer_reach(rec):
    env = rec.env
    place(env, (2.11, -3.70), math.pi / 2)
    env.data.qpos[env.doors["laundry_door"]["q"]] = math.radians(95)
    env.data.qpos[env.mdoor["dryer"][0]] = math.radians(100)
    mujoco.mj_forward(env.model, env.data)
    rec.fixed("laundry")
    rec.caption = "Reach into the dryer drum (door centre 1.55 m): lift up, wrist out, slide in"
    q = B.STOW_Q.copy()
    q[1] = math.radians(90)
    q[0] = 0.0
    B.run(env, 11, lambda e: B.command(e, lift=B.lift_cmd(e, 1.47), servos=B.servo_duty(e, q)))
    s = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "in_dryer")
    B.run(env, 5, lambda e: B.command(e, lift=B.lift_cmd(e, 1.47), arm=B.arm_cmd(e, 0.54), servos=B.servo_duty(e, q)))
    g = B.grip_point(env)
    c, R, hs = env.data.site_xpos[s], env.data.site_xmat[s].reshape(3, 3), env.model.site_size[s]
    inside = bool(np.all(np.abs(R.T @ (g - c)) <= hs + 0.02))
    rec.hold(1.5)
    return inside, (f"Gripper inside the dryer drum at {g[2]:.2f} m." if inside else
                    f"Gripper reached {g[2]:.2f} m but not inside the drum.")


def task_shove(rec):
    env = rec.env
    place(env, (2.1, -1.4), 0.0)
    rec.follow(3.0, -12, 90, 0.8)
    rec.caption = "Lift up, arm out. Someone bumps the mast top: 80 N for 0.2 s"
    B.run(env, 10, lambda e: B.command(e, lift=B.lift_cmd(e, 1.4), arm=B.arm_cmd(e, 0.5)))
    env.push = (np.array([0.0, 80.0, 0.0]), env.data.xpos[env.head_body].copy())
    env.push_left = 0.2
    info = B.run(env, 3, lambda e: B.command(e))
    survived = not info["tipped"]
    rec.caption = "Now a hard shove: 200 N for 0.2 s"
    env.push = (np.array([0.0, 200.0, 0.0]), env.data.xpos[env.head_body].copy())
    env.push_left = 0.2
    info = B.run(env, 4, lambda e: B.command(e))
    tipped = info["tipped"]
    return survived, (f"Rode out the 80 N bump; the 200 N shove {'tipped it over' if tipped else 'did not tip it'}. "
                      "A tipped robot stops its motors and calls for help.")


TASKS = {
    "drive": ("Drive the hallway", task_drive),
    "pick_can": ("Pick up an empty can", task_pick("can_1", "can")),
    "full_can": ("Don't spill the drink", task_full_can),
    "pick_sneaker": ("Pick up a sneaker", task_pick("sneaker_l", "sneaker")),
    "pick_sock": ("Pick up a sock", task_pick("sock_1", "sock")),
    "pick_paper_bag": ("Carry a paper bag", task_carry_bag),
    "door": ("Open the laundry closet", task_door),
    "dryer": ("Reach into the dryer", task_dryer_reach),
    "dock": ("Charge itself", task_dock),
    "shove": ("Take a bump", task_shove),
}


def main():
    names = sys.argv[1:] or list(TASKS)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(ROOT, "docs", "tasks.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    for name in names:
        title, fn = TASKS[name]
        t = time.time()
        env = lab()
        rec = Recorder(env, os.path.join(OUT, f"{name}.mp4"))
        ok, note = fn(rec)
        rec.close()
        results[name] = {"title": title, "success": bool(ok), "note": note, "video": f"videos/{name}.mp4",
                         "seconds": round(env.t, 1)}
        print(f"{'OK  ' if ok else 'FAIL'} {name}: {note} ({time.time() - t:.0f} s to record)")
    ordered = {k: results[k] for k in TASKS if k in results}
    json.dump(ordered, open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
