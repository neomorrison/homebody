"""HomebodyEnv: one robot in a room, alive only as long as its battery.

Actions are muscle activations: one PWM duty cycle in [-1, 1] per servo, the
same number the physical robot writes to each Feetech servo in PWM mode. There
are no "walk" or "sleep" commands. Standing still costs energy; so does falling.

Observations are only what the physical robot's own sensors report (see
``spec.SensorSpec``). Ground truth (position, velocity, distance to the dock,
damage) goes in ``info`` for analysis, critics and reward shaping, and never
into the observation.

The episode ends when the robot dies: its battery hits the BMS cutoff, or
impacts have broken it. It does not end when it falls over; it can get up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np

from . import power as P
from . import spec as S
from . import world_xml as W

DEFAULT_POSE = {   # radians, a slight crouch; used for resets only
    "hip_pitch_l": -0.35, "knee_l": 0.7, "ankle_pitch_l": -0.35,
    "hip_pitch_r": -0.35, "knee_r": 0.7, "ankle_pitch_r": -0.35,
    "shoulder_roll_l": 0.15, "shoulder_roll_r": -0.15, "elbow_l": -0.4, "elbow_r": -0.4,
}


@dataclass
class EnvConfig:
    capacity_scale: float = 1.0          # <1 shortens the battery for faster lifetimes in training
    soc_range: tuple = (0.3, 1.0)        # initial state of charge
    max_seconds: float = 3600.0          # time limit (truncation, not death)
    start: str = "standing"              # "standing" | "lying" | "random"
    randomize_layout: bool = True
    randomize_physics: bool = True
    sensor_noise: bool = True
    camera: tuple | None = None          # (height, width) to add the head camera image
    damage_force: float = 80.0           # N, impacts above this on non-foot parts cause damage
    damage_limit: float = 40.0           # N s of excess impulse before something breaks
    reward: str = "survival"             # see REWARDS
    seed_layout_every_reset: bool = True
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# rewards: functions of (info, prev_info). The env does not decide what the
# robot should want; pick one, or pass a callable.
# ----------------------------------------------------------------------------

def r_survival(info, prev):
    """Only staying alive counts. Death is the end of all future reward."""
    return 0.0 if info["dead"] else 1.0 / S.CONTROL_HZ


def r_energy(info, prev):
    """Survival plus the change in stored energy: eating is good, waste is bad."""
    return r_survival(info, prev) + 0.02 * (info["soc"] - prev["soc"]) * S.CONTROL_HZ * 100


def r_dock_shaped(info, prev):
    """Survival + energy + progress toward the dock + staying upright. A teaching aid."""
    r = r_energy(info, prev)
    r += 2.0 * (prev["dock_dist"] - info["dock_dist"])
    r += 0.02 * info["upright"]
    r += 0.05 * info["charging"]
    return r


REWARDS = {"survival": r_survival, "energy": r_energy, "dock_shaped": r_dock_shaped}


class HomebodyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": S.CONTROL_HZ}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.model = W.build_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_sub = S.PHYSICS_HZ // S.CONTROL_HZ
        self.dt = 1.0 / S.CONTROL_HZ

        idj = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        idb = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        idg = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        ids = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
        self.qadr = np.array([m.jnt_qposadr[idj(n)] for n in S.JOINT_NAMES])
        self.vadr = np.array([m.jnt_dofadr[idj(n)] for n in S.JOINT_NAMES])
        self.robot = idb("robot")
        self.dock = idb("dock")
        self.furniture = [idb(f"furniture_{k}") for k in range(W.ROOM.n_furniture)]
        self.rug = idb("rug")
        self.sole = {s: idg(f"sole_{s}") for s in "lr"}
        self.plate = {"a": idg("dock_plate_a"), "b": idg("dock_plate_b")}
        self.beacon = ids("beacon")
        self.dock_target = ids("dock_target")
        self.ir_sites = [ids(f"ir_{k}") for k in ("front", "left", "back", "right")]
        self.imu = ids("imu")
        # robot geoms (for damage) = everything in the robot subtree
        root_of = m.body_rootid
        self.robot_geom = np.array([root_of[m.geom_bodyid[g]] == self.robot for g in range(m.ngeom)])
        self.hand_geoms = {idg("hand_l"), idg("hand_r")}
        self.foot_geoms = set(self.sole.values())
        self.sensor_slice = {}
        for name in ["imu_acc", "imu_gyro", "imu_quat"] + [f"fsr_{s}_{k}" for s in "lr" for k in range(4)]:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
            a = m.sensor_adr[sid]
            self.sensor_slice[name] = slice(a, a + m.sensor_dim[sid])

        self.stall = np.array([j.servo.stall_torque for j in S.JOINTS])
        self.vrated = np.array([j.servo.v_rated for j in S.JOINTS])
        self.nominal = {
            "body_mass": m.body_mass.copy(), "geom_friction": m.geom_friction.copy(),
            "gainprm": m.actuator_gainprm[:, 0].copy(),
        }
        self.default_q = np.array([DEFAULT_POSE.get(n, 0.0) for n in S.JOINT_NAMES])

        self.battery = P.Battery(capacity_scale=self.cfg.capacity_scale)
        self.charger = P.Charger()
        self.servos = P.ServoBank()

        self.ray_groups = np.array([1, 1, 0, 0, 1, 1], np.uint8)   # skip robot (2) and the LED (3)
        self._renderer = None
        self._eye = None

        n_obs = len(self._obs_names())
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        if self.cfg.camera:
            h, w = self.cfg.camera
            self.observation_space = gym.spaces.Dict({
                "proprio": self.observation_space,
                "eye": gym.spaces.Box(0, 255, (h, w, 3), np.uint8),
            })
        self.action_space = gym.spaces.Box(-1.0, 1.0, (S.N_JOINTS,), np.float32)
        self.reward_fn = REWARDS[self.cfg.reward] if isinstance(self.cfg.reward, str) else self.cfg.reward

    # ------------------------------------------------------------------ names
    @staticmethod
    def _obs_names():
        J = S.JOINT_NAMES
        return ([f"q.{j}" for j in J] + [f"dq.{j}" for j in J] + [f"load.{j}" for j in J]
                + [f"temp.{j}" for j in J]
                + ["grav.x", "grav.y", "grav.z", "gyro.x", "gyro.y", "gyro.z", "acc.x", "acc.y", "acc.z"]
                + [f"fsr.{s}{k}" for s in "lr" for k in range(4)]
                + ["batt.v", "batt.i", "batt.soc", "charging"]
                + ["ir.front", "ir.left", "ir.back", "ir.right"]
                + [f"prev_act.{j}" for j in J])

    obs_names = property(lambda self: self._obs_names())

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m, d, rng = self.model, self.data, self.np_random
        mujoco.mj_resetData(m, d)
        if self.cfg.randomize_layout:
            self._randomize_layout()
        if self.cfg.randomize_physics:
            self._randomize_physics()
        self._place_robot()
        self.battery.reset(rng.uniform(*self.cfg.soc_range))
        self.charger.reset()
        self.servos.reset(ambient=rng.uniform(18, 30))
        self.prev_action = np.zeros(S.N_JOINTS)
        self.damage = 0.0
        self.t = 0.0
        self.energy_used = 0.0
        self.energy_charged = 0.0
        self._set_voltage(self.battery.v)
        mujoco.mj_forward(m, d)
        self.info = self._info()
        return self._obs(), dict(self.info)

    def _randomize_layout(self):
        m, rng = self.model, self.np_random
        X, Y = W.ROOM.size[0] / 2, W.ROOM.size[1] / 2
        # dock against a random wall, facing into the room
        wall = rng.integers(4)
        back = S.DOCK.base_size[0] / 2 + 0.01
        along = rng.uniform(-0.7, 0.7)
        pos, yaw = {
            0: ((X - back, along * Y), math.pi),
            1: ((-X + back, along * Y), 0.0),
            2: ((along * X, Y - back), -math.pi / 2),
            3: ((along * X, -Y + back), math.pi / 2),
        }[int(wall)]
        # body x axis points into the room: the post is at local -x, i.e. at the wall
        self._set_body(self.dock, pos, yaw)
        taken = [(np.array(pos), 0.6)]
        for b in self.furniture:
            g = m.body_geomadr[b]
            r = float(np.hypot(*m.geom_size[g][:2]))
            for _ in range(50):
                p = np.array([rng.uniform(-X + r, X - r), rng.uniform(-Y + r, Y - r)])
                if all(np.linalg.norm(p - q) > r + rq + 0.3 for q, rq in taken):
                    break
            taken.append((p, r))
            self._set_body(b, p, rng.uniform(-math.pi, math.pi))
        self._set_body(self.rug, (rng.uniform(-X + 1, X - 1), rng.uniform(-Y + 0.7, Y - 0.7)), rng.uniform(0, math.pi))
        self._taken = taken

    def _set_body(self, b, xy, yaw):
        self.model.body_pos[b][:2] = xy
        self.model.body_quat[b] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    def _randomize_physics(self):
        m, rng, n = self.model, self.np_random, self.nominal
        m.body_mass[:] = n["body_mass"] * rng.uniform(0.9, 1.1, m.nbody)
        m.geom_friction[:] = n["geom_friction"]
        m.geom_friction[:, 0] *= rng.uniform(0.75, 1.25)
        self.strength = rng.uniform(0.85, 1.05, S.N_JOINTS)   # servo-to-servo spread

    def _place_robot(self):
        m, d, rng = self.model, self.data, self.np_random
        X, Y = W.ROOM.size[0] / 2, W.ROOM.size[1] / 2
        taken = getattr(self, "_taken", [(np.array([1.0, 0.0]), 0.6)])
        for _ in range(100):
            p = np.array([rng.uniform(-X + 0.4, X - 0.4), rng.uniform(-Y + 0.4, Y - 0.4)])
            if all(np.linalg.norm(p - q) > r + 0.4 for q, r in taken):
                break
        yaw = rng.uniform(-math.pi, math.pi)
        mode = self.cfg.start if self.cfg.start != "random" else rng.choice(["standing", "lying"])
        q = d.qpos
        q[0:2] = p
        q[self.qadr] = self.default_q + rng.normal(0, 0.03, S.N_JOINTS)
        if mode == "standing":
            q[2] = S.GEOM.standing_pelvis_height * 0.97 - 0.02
            q[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        else:   # lying on its back or front
            q[2] = 0.08
            pitch = rng.choice([-1, 1]) * math.pi / 2
            qy = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            qp = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
            out = np.empty(4)
            mujoco.mju_mulQuat(out, qy, qp)
            q[3:7] = out
        lo, hi = m.jnt_range[[self._jid(n) for n in S.JOINT_NAMES]].T
        q[self.qadr] = np.clip(q[self.qadr], lo, hi)
        if mode == "standing":   # drop the robot so the lower sole rests 2 mm above the floor
            mujoco.mj_kinematics(m, d)
            low = min(d.geom_xpos[g][2] - m.geom_size[g][2] for g in self.sole.values())
            q[2] += 0.002 - low

    def _jid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    # ------------------------------------------------------------------- step
    def _set_voltage(self, v):
        v = max(v, 0.0)
        self.model.actuator_gainprm[:, 0] = self.nominal["gainprm"] * (v / self.vrated) * getattr(self, "strength", 1.0)
        self.model.actuator_forcerange[:, 0] = -self.stall * np.minimum(1.0, v / self.vrated) * 1.0
        self.model.actuator_forcerange[:, 1] = self.stall * np.minimum(1.0, v / self.vrated) * 1.0

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.battery.dead:
            a = np.zeros_like(a)                      # BMS open: everything is limp
        a = np.where(self.servos.tripped, 0.0, a)      # overheated servos cut out
        self._set_voltage(self.battery.v)
        d.ctrl[:] = a

        load_i, impact = 0.0, 0.0
        docked_frac = 0.0
        sub_dt = 1.0 / S.PHYSICS_HZ
        for _ in range(self.n_sub):
            mujoco.mj_step(m, d)
            torque = d.actuator_force.copy()
            omega = d.qvel[self.vadr]
            load_i += self.servos.step(torque, a, omega, sub_dt) / self.n_sub
            impact += self._impacts() * sub_dt
            docked_frac += self._docked() / self.n_sub

        docked = docked_frac > 0.5
        load_i += S.ELECTRONICS.idle_power(S.N_JOINTS) / max(self.battery.v, 1.0)
        charge_i = self.charger.current(docked, self.battery, self.dt)
        e = self.battery.step(0.0 if self.battery.dead else load_i, charge_i, self.dt)
        self.energy_used += max(e, 0.0)
        self.energy_charged += max(-e, 0.0)
        self.damage += impact
        self.prev_action = a
        self.t += self.dt

        prev = self.info
        self.info = self._info(charging=charge_i > 0, load_i=load_i, charge_i=charge_i)
        terminated = bool(self.info["dead"])
        truncated = self.t >= self.cfg.max_seconds
        reward = float(self.reward_fn(self.info, prev))
        return self._obs(), reward, terminated, truncated, dict(self.info)

    def _impacts(self):
        """Excess normal force on any robot part except feet and hands hitting something."""
        d = self.data
        total = 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            for g, other in ((g1, g2), (g2, g1)):
                if self.robot_geom[g] and not self.robot_geom[other] and g not in self.foot_geoms \
                        and g not in self.hand_geoms:
                    mujoco.mj_contactForce(self.model, d, i, f6)
                    total += max(f6[0] - self.cfg.damage_force, 0.0)
        return total

    def _docked(self) -> bool:
        """Each sole touches a different plate, and no sole bridges both (the dock would refuse)."""
        d = self.data
        touch = {s: set() for s in "lr"}
        for i in range(d.ncon):
            c = d.contact[i]
            for s, sg in self.sole.items():
                for p, pg in self.plate.items():
                    if {c.geom1, c.geom2} == {sg, pg}:
                        touch[s].add(p)
        l, r = touch["l"], touch["r"]
        return len(l) == 1 and len(r) == 1 and l != r

    # -------------------------------------------------------------- sensing
    def _ir(self):
        m, d = self.model, self.data
        bpos = d.site_xpos[self.beacon]
        out = np.zeros(4)
        geomid = np.zeros(1, np.int32)
        for k, s in enumerate(self.ir_sites):
            p = d.site_xpos[s]
            v = bpos - p
            dist = float(np.linalg.norm(v))
            u = v / max(dist, 1e-6)
            facing = d.site_xmat[s].reshape(3, 3)[:, 0]
            c = float(facing @ u)
            if c <= 0:
                continue
            hit = mujoco.mj_ray(m, d, p, u, self.ray_groups, 1, -1, geomid)
            if 0 <= hit < dist - 0.02:
                continue           # something solid in the way
            # beacon LED has a wide (~120 deg) emission cone toward the room
            bx = d.site_xmat[self.beacon].reshape(3, 3)[:, 0]
            emit = max(float(bx @ -u), 0.0) ** 0.5
            out[k] = S.DOCK.ir_beacon_power * emit * c ** 2 / max(dist * dist, 0.04)
        return out

    def _obs(self):
        d, ns, rng = self.data, self.cfg.sensor_noise, self.np_random
        sn = S.SENSORS
        noise = (lambda sd, n: rng.normal(0, sd, n)) if ns else (lambda sd, n: 0.0)
        q = d.qpos[self.qadr] + noise(sn.enc_noise, S.N_JOINTS)
        dq = d.qvel[self.vadr] + noise(sn.vel_noise, S.N_JOINTS)
        load = d.actuator_force / self.stall
        temp = (self.servos.temp - 25.0) / 50.0
        quat = d.sensordata[self.sensor_slice["imu_quat"]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        grav = R.reshape(3, 3).T @ np.array([0, 0, -1.0]) + noise(sn.imu_grav_noise, 3)
        gyro = d.sensordata[self.sensor_slice["imu_gyro"]] + noise(sn.imu_gyro_noise, 3)
        acc = d.sensordata[self.sensor_slice["imu_acc"]] / 9.81 + noise(sn.imu_acc_noise / 9.81, 3)
        fsr = np.array([d.sensordata[self.sensor_slice[f"fsr_{s}_{k}"]][0] for s in "lr" for k in range(4)])
        fsr = fsr + noise(sn.fsr_noise, 8)
        b = self.battery
        power = np.array([(b.v - 11.1) / 1.5 + (noise(sn.v_noise, 1)[0] if ns else 0.0),
                          b.i / 5.0 + (noise(sn.i_noise, 1)[0] if ns else 0.0),
                          b.soc, float(self.info.get("charging", 0.0)) if hasattr(self, "info") else 0.0])
        ir = np.log1p(self._ir() * 10.0)
        vec = np.concatenate([q, dq / 10.0, load, temp, grav, gyro / 5.0, acc, fsr / 15.0, power, ir,
                              self.prev_action]).astype(np.float32)
        if self.cfg.camera:
            return {"proprio": vec, "eye": self._eye_image()}
        return vec

    def _eye_image(self):
        h, w = self.cfg.camera
        if self._eye is None:
            self._eye = mujoco.Renderer(self.model, h, w)
        self._eye.update_scene(self.data, camera="eye")
        return self._eye.render()

    def _info(self, charging=False, load_i=0.0, charge_i=0.0):
        d = self.data
        torso = d.xpos[self.robot]
        up = d.xmat[self.robot].reshape(3, 3)[2, 2]
        target = d.site_xpos[self.dock_target]
        return {
            "t": self.t, "soc": self.battery.soc, "v": self.battery.v, "load_a": load_i,
            "charge_a": charge_i, "charging": bool(charging),
            "dead": bool(self.battery.dead or self.damage > self.cfg.damage_limit),
            "battery_dead": bool(self.battery.dead), "damage": self.damage,
            "broken": bool(self.damage > self.cfg.damage_limit),
            "pos": torso.copy(), "height": float(torso[2]), "upright": float(up),
            "fallen": bool(up < 0.5 or torso[2] < 0.15),
            "dock_dist": float(np.linalg.norm((target - torso)[:2])),
            "vel": d.qvel[0:3].copy(), "servo_temp_max": float(self.servos.temp.max()),
            "energy_used_wh": self.energy_used / 3600.0, "energy_charged_wh": self.energy_charged / 3600.0,
        }

    # ------------------------------------------------------------- render
    def render(self, camera="overview", width=960, height=540):
        if self._renderer is None or self._renderer.width != width:
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        for r in (self._renderer, self._eye):
            if r is not None:
                r.close()
