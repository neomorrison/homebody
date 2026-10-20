"""HomebodyEnv: the robot in the owner's home, alive only as long as its battery.

Actions are the numbers the robot's motor drivers take (``spec.ACTION_NAMES``):
duty cycles for the two hub motors and the six bus servos, and speed commands
for the lift and telescope steppers. There are no "go to", "grasp", "open the
door" or "dock" commands.

Observations are what the robot's own sensors report (``obs_names``): wheel
speeds, drifting dead-reckoned odometry, stepper positions, servo encoders,
loads and temperatures, IMU, a 240 degree lidar scan, cliff sensors, bumper,
battery monitor, charging flag, dock beacon and part self-diagnostics. Head
and palm camera images are added with ``EnvConfig(cameras=(h, w))``. Ground
truth goes in ``info`` only.

The world fights back: the floor is uneven with rugs, sills and cables; the
desk chair rolls away when bumped; the laundry door latches until its lever is
pressed; machine doors are held shut by a catch; people bump the robot
(``push_rate``); hard knocks break parts (``spec.PARTS``). The episode ends
when the robot dies: the battery hits the BMS cutoff, it tips over, or its
wrist or mast breaks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np

from . import home as H
from . import items as I
from . import power as P
from . import spec as S
from . import terrain as T
from . import world_xml as W

STOW = {"wrist_yaw": 0.0, "wrist_pitch": 0.0, "wrist_roll": 0.0, "gripper": 0.0, "head_pan": 0.0, "head_tilt": 0.6}
STOW_LIFT = 0.25       # m of lift travel: gripper clear of the wheels and the floor
TERMINAL_PARTS = ("wrist", "mast")


@dataclass
class EnvConfig:
    capacity_scale: float = 1.0          # <1 shortens the battery for faster lifetimes in training
    soc_range: tuple = (0.3, 1.0)
    max_seconds: float = 1800.0
    terrain_level: float = 1.0           # 0 lab floor, 1 ordinary home, 2+ harder (terrain.py)
    kinds: tuple = tuple(I.KINDS)        # which of the owner's clutter to put out (see items.py)
    scatter: bool = False                # clutter anywhere on free floor, not where it was in the scan
    item_dropout: float = 0.0            # leave each item out with this probability
    start_docked: float = 0.0            # probability of starting on the charger
    start_room: tuple = ("bedroom", "hallway", "living", "kitchen")
    randomize_look: bool = True          # vary colours, lighting and the movable furniture each episode
    push_rate: float = 0.0               # accidental shoves per minute
    push_force: tuple = (20.0, 80.0)     # N, applied near the top of the mast for 0.2 s
    randomize_physics: bool = True
    sensor_noise: bool = True
    cameras: tuple | None = None         # (h, w): add head and palm RGB images
    depth: bool = False                  # with cameras: add the optional depth camera's image
    servo_mode: str = S.SERVO_MODE       # "position" (servo's own loop, brain sends targets) or "pwm" (raw duty)
    reward: object = "survival"          # name in REWARDS or a callable(info, prev)


# ----------------------------------------------------------------------------
# rewards: functions of (info, prev_info)
# ----------------------------------------------------------------------------

def r_survival(info, prev):
    return 0.0 if info["dead"] else 1.0 / S.CONTROL_HZ


def r_energy(info, prev):
    return r_survival(info, prev) + 2.0 * (info["soc"] - prev["soc"])


def r_chores(info, prev):
    """Survival + energy + one point per item put where it belongs (minus one per item taken out)."""
    return r_energy(info, prev) + float(info["tidy"] - prev["tidy"])


def r_dock_shaped(info, prev):
    r = r_energy(info, prev)
    if info["soc"] < 0.5:
        r += 1.0 * (prev["dock_dist"] - info["dock_dist"])
    return r + 0.01 * info["charging"]


REWARDS = {"survival": r_survival, "energy": r_energy, "chores": r_chores, "dock_shaped": r_dock_shaped}


class HomebodyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": S.CONTROL_HZ}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.model = W.build_model(self.cfg.servo_mode)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_sub = S.PHYSICS_HZ // S.CONTROL_HZ
        self.dt = 1.0 / S.CONTROL_HZ

        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        Gm = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        Si = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
        qa = lambda n: m.jnt_qposadr[J(n)]
        va = lambda n: m.jnt_dofadr[J(n)]
        self.robot = B("robot")
        self.rq, self.rv = qa("root"), va("root")
        self.wheel_v = np.array([va(w) for w in S.WHEELS])
        self.servo_q = np.array([qa(n) for n in S.SERVO_NAMES])
        self.servo_v = np.array([va(n) for n in S.SERVO_NAMES])
        self.lift_q, self.lift_v = qa("lift"), va("lift")
        self.arm_q = np.array([qa(f"arm_{k}") for k in range(1, S.GEOM.n_stages + 1)])
        self.arm_v = np.array([va(f"arm_{k}") for k in range(1, S.GEOM.n_stages + 1)])
        self.act = {n: i for i, n in enumerate(S.ACTION_NAMES)}
        self.duty_act = np.array([self.act[n] for n in S.DUTY_NAMES])
        self.axis_act = np.array([self.act["lift"], self.act["arm"]])
        self.pads = (Gm("pad_l"), Gm("pad_r"))
        self.contacts = (Gm("dock_contact_a"), Gm("dock_contact_b"))
        self.beacon, self.dock_target = Si("beacon"), Si("dock_target")
        self.lidar_site = Si("lidar")
        self.cliff_sites = [Si(f"cliff_{k}") for k in ("fl", "fr", "rl", "rr")]
        self.ir_sites = [Si(f"ir_{k}") for k in ("front", "left", "back", "right")]
        self.head_body = B("head_tilt")
        self.doors = {d.name: dict(spec=d, q=qa(d.name), v=va(d.name), jid=J(d.name), range=m.jnt_range[J(d.name)].copy(),
                                   lever=qa(f"{d.name}_lever_a") if d.latch == "lever" else None) for d in H.DOORS}
        self.mdoor = {n: (qa(f"{n}_door"), va(f"{n}_door")) for n in ("washer", "dryer")}
        self.chair_q, self.chair_v = qa("chair"), va("chair")
        self.bins = {b.name: B(b.name) for b in H.BINS}
        containers = {b.name: Si(f"in_{b.name}") for b in H.BINS}
        containers.update({name: Si(f"in_{name}") for name, _, _ in H.ZONES})
        containers.update({"washer": Si("in_washer"), "dryer": Si("in_dryer"), "tote": Si("tote")})
        self.items = I.Items(m, containers)
        self.terrain = T.Terrain(m, H.EXTENT)
        self.look = W.Look(m, H)
        self.terrain_state = None

        root_of = m.body_rootid
        self.robot_geom = np.array([root_of[m.geom_bodyid[g]] == self.robot for g in range(m.ngeom)])
        self.part_of = np.full(m.ngeom, -1)
        part_geoms = {
            "wrist": ["servo_wrist_yaw", "servo_wrist_pitch", "servo_wrist_roll", "servo_gripper", "finger_l", "finger_r"],
            "palm_cam": ["palm"],
            "head": ["head", "servo_head_pan", "mast_cap"],
            "lidar": ["lidar"],
            "mast": ["mast", "carriage", "stage_1", "stage_2", "stage_3"],
        }
        self.part_names = [p.name for p in S.PARTS]
        self.part_thresh = np.array([p.force_threshold for p in S.PARTS])
        self.part_limit = np.array([p.impulse_limit for p in S.PARTS])
        for k, p in enumerate(S.PARTS):
            for g in part_geoms[p.name]:
                self.part_of[Gm(g)] = k
        self.sensor_slice = {}
        for name in ("imu_acc", "imu_gyro", "imu_quat", "bumper"):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self.sensor_slice[name] = slice(m.sensor_adr[sid], m.sensor_adr[sid] + m.sensor_dim[sid])

        self.stall = np.array([mo.stall_torque for mo in S.DUTY_MOTORS])
        self.vrated = np.array([mo.v_rated for mo in S.DUTY_MOTORS])
        self.on_battery = np.array([mo.supply == "battery" for mo in S.DUTY_MOTORS])
        self.is_servo = np.array([n in S.SERVO_NAMES for n in S.DUTY_NAMES])
        self.torque_limit = np.array([next((j.torque_limit for j in S.SERVOS if j.name == n), 1.0) for n in S.DUTY_NAMES])
        self.position_mode = (self.cfg.servo_mode == "position") & self.is_servo
        self.wrist_duty = np.array([S.DUTY_NAMES.index(j.name) for j in S.WRIST])
        self.head_duty = np.array([S.DUTY_NAMES.index(j.name) for j in S.HEAD])
        self.nominal = {"body_mass": m.body_mass.copy(), "gainprm": m.actuator_gainprm[:, 0].copy(),
                        "axis_force": m.actuator_forcerange[self.axis_act].copy()}
        carriage = B("carriage")
        self.counterbalance = float(m.body_subtreemass[carriage]) * S.G_ACC
        self.stow = np.array([STOW[n] for n in S.SERVO_NAMES])
        self.strength = np.ones(len(S.DUTY_MOTORS))
        self.tyre_scale, self.odom_bias = 1.0, np.zeros(2)

        self.battery = P.Battery(S.BATTERY, capacity_scale=self.cfg.capacity_scale)
        self.charger = P.Charger(S.DOCK, S.BATTERY)
        self.motors = P.MotorBank(S.DUTY_MOTORS)

        L = S.LIDAR
        a = np.radians(np.linspace(-L.fov_deg / 2, L.fov_deg / 2, L.rays))
        self.lidar_dirs = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1)
        self.ray_groups = np.array([1, 1, 0, 0, 1, 1], np.uint8)
        self._renderer = None
        self._cam = {}

        vec = gym.spaces.Box(-np.inf, np.inf, (len(self.obs_names),), np.float32)
        if self.cfg.cameras:
            h, w = self.cfg.cameras
            spaces = {"state": vec, "head_rgb": gym.spaces.Box(0, 255, (h, w, 3), np.uint8),
                      "palm_rgb": gym.spaces.Box(0, 255, (h, w, 3), np.uint8)}
            if self.cfg.depth:
                spaces["head_depth"] = gym.spaces.Box(0, np.inf, (h, w), np.float32)
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            self.observation_space = vec
        self.action_space = gym.spaces.Box(-1.0, 1.0, (S.N_ACTIONS,), np.float32)
        self.reward_fn = REWARDS[self.cfg.reward] if isinstance(self.cfg.reward, str) else self.cfg.reward

    # ------------------------------------------------------------------ names
    @property
    def obs_names(self):
        Sv, D = S.SERVO_NAMES, S.DUTY_NAMES
        return (["wheel_vel.l", "wheel_vel.r", "odom.x", "odom.y", "odom.sin_yaw", "odom.cos_yaw",
                 "lift.pos", "lift.vel", "arm.pos", "arm.vel"]
                + [f"q.{j}" for j in Sv] + [f"dq.{j}" for j in Sv]
                + [f"load.{j}" for j in D] + [f"temp.{j}" for j in D]
                + ["grav.x", "grav.y", "grav.z", "gyro.x", "gyro.y", "gyro.z", "acc.x", "acc.y", "acc.z"]
                + [f"lidar.{k}" for k in range(S.LIDAR.rays)]
                + ["cliff.fl", "cliff.fr", "cliff.rl", "cliff.rr", "bumper"]
                + ["batt.v", "batt.i", "batt.soc", "charging"]
                + ["ir.front", "ir.left", "ir.back", "ir.right"]
                + [f"ok.{p}" for p in self.part_names]
                + [f"prev_act.{a}" for a in S.ACTION_NAMES])

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m, d, rng = self.model, self.data, self.np_random
        mujoco.mj_resetData(m, d)
        if self.cfg.randomize_physics:
            m.body_mass[:] = self.nominal["body_mass"] * rng.uniform(0.9, 1.1, m.nbody)
            self.strength = rng.uniform(0.85, 1.05, len(S.DUTY_MOTORS))
            self.tyre_scale = rng.uniform(0.85, 1.1)
            self.odom_bias = rng.normal(0, S.NOISE.odom_drift, 2)
        self.part_damage = np.zeros(len(S.PARTS))
        self.broken = np.zeros(len(S.PARTS), bool)
        self._layout()
        if rng.uniform() < self.cfg.start_docked:
            self.put_on_dock()
        else:
            self._place_robot()
        self._place_chair_and_doors()
        rects = self._obstacles() + [(*d.qpos[self.rq:self.rq + 2], 0.35, 0.35, 0.0)]
        free = lambda p, r: H.is_free(p, r, rects, for_items=True)
        mujoco.mj_kinematics(m, d)
        bucket = d.xpos[self.bins["pod_bucket"]].copy()
        self.items.place(rng, self.cfg.kinds, d.qpos, d.qvel, self.terrain.height_at, free, bucket,
                         scatter=self.cfg.scatter, dropout=self.cfg.item_dropout)
        self.battery.reset(rng.uniform(*self.cfg.soc_range))
        self.charger.reset()
        self.motors.reset(ambient=rng.uniform(18, 30))
        self.prev_action = np.zeros(S.N_ACTIONS)
        self.t = 0.0
        self.energy_used = 0.0
        self.energy_charged = 0.0
        self.odom = np.zeros(3)
        self.push_left, self.push = 0.0, None
        self._apply_supply()
        self._latch()
        mujoco.mj_forward(m, d)
        self.info = self._info()
        info = dict(self.info)
        info["terrain"] = self.terrain_state
        return self._obs(), info

    def _obstacles(self):
        (x, y), yaw = H.DOCK_POSE
        extra = [(x + 0.15 * math.cos(yaw), y + 0.15 * math.sin(yaw), 0.15, 0.2, yaw),
                 (*H.CHAIR_HOME, 0.4, 0.4, 0.0)]
        return H.obstacle_rects(extra)

    def _layout(self):
        m, rng, lvl = self.model, self.np_random, self.cfg.terrain_level
        tr = self.terrain
        tr.generate_heights(rng, lvl)
        self.look.randomize(rng if self.cfg.randomize_look else None)
        fl = lambda c, half, yaw=0.0, pad=0.06: tr.flatten(c[0], c[1], yaw, half[0], half[1], pad)
        for name in ("washer", "dryer"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            mc = H.WASHER if name == "washer" else H.DRYER
            m.body_pos[b][2] = fl(H.WASHER.center, (H.WASHER.half[1], H.WASHER.half[0])) + mc.base_z
        (x, y), yaw = H.DOCK_POSE
        c, s = math.cos(yaw), math.sin(yaw)
        m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "dock")][2] = tr.flatten(
            x + 0.3 * c, y + 0.3 * s, yaw, 0.35, 0.3)
        for bdef in H.BINS:
            b = self.bins[bdef.name]
            p = np.array(bdef.center, float)
            m.body_pos[b][:2] = p
            m.body_pos[b][2] = bdef.z if bdef.z else fl(p, (bdef.inner[0] + 0.01, bdef.inner[1] + 0.01))
        rects = self._obstacles()
        open_floor = {r: H.ROOMS[r] for r in ("living", "hallway", "kitchen", "foyer")}
        sills = [(np.array(xy), yaw) for xy, yaw in H.DOORWAYS.values()]
        self.terrain_state = tr.randomize(
            rng, lvl, self._set_body, lambda p, r: H.is_free(p, r, rects),
            sill_spots=sills, patch_free=lambda p, r: H.is_free(p, r, rects, open_floor),
            floor_choices=("laminate", "hardwood"), tyre_scale=self.tyre_scale)
        tr.upload()

    def _set_body(self, b, xy, yaw):
        self.model.body_pos[b][:2] = xy
        self.model.body_quat[b] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    def _floor_under(self, xy, yaw):
        g = S.GEOM
        c, s = math.cos(yaw), math.sin(yaw)
        pts = [(0, g.wheel_track / 2), (0, -g.wheel_track / 2), (g.caster_x, 0), (-g.caster_x, 0)]
        return max(self.terrain.height_at(xy[0] + c * px - s * py, xy[1] + s * px + c * py) for px, py in pts)

    def _pose_robot(self, xy, yaw):
        d, g = self.data, S.GEOM
        q = d.qpos
        q[self.rq:self.rq + 2] = xy
        q[self.rq + 3:self.rq + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        q[self.rq + 2] = self._floor_under(xy, yaw) + (g.base_bottom + g.base_top) / 2 + 0.004
        q[self.servo_q] = self.stow
        q[self.lift_q] = STOW_LIFT
        q[self.arm_q] = 0.0
        d.qvel[self.rv:self.rv + 6] = 0

    def _place_robot(self):
        rng = self.np_random
        rects = self._obstacles()
        rooms = {r: H.ROOMS[r] for r in self.cfg.start_room}
        p = np.array(H.A(-1.0, 0.5))
        X, Y = H.EXTENT[0] / 2, H.EXTENT[1] / 2
        for _ in range(400):
            q = np.array([rng.uniform(-X, X), rng.uniform(-Y, Y)])
            if H.is_free(q, 0.48, rects, rooms):
                p = q
                break
        self._pose_robot(p, rng.uniform(-math.pi, math.pi))

    def put_on_dock(self):
        """Back the robot against the dock contacts (for curricula and tests)."""
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        (x, y), yaw = H.DOCK_POSE
        self._pose_robot(d.site_xpos[self.dock_target][:2].copy(), yaw)
        mujoco.mj_forward(m, d)

    def _place_chair_and_doors(self):
        d, rng = self.data, self.np_random
        x, y = H.CHAIR_HOME
        cx, cy = x + rng.uniform(-0.2, 0.2), y + rng.uniform(-0.25, 0.1)
        yaw = rng.uniform(-math.pi, math.pi)
        d.qpos[self.chair_q:self.chair_q + 7] = [cx, cy, self.terrain.height_at(cx, cy) + 0.012,
                                                 math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qvel[self.chair_v:self.chair_v + 6] = 0
        for dd in self.doors.values():
            spec = dd["spec"]
            d.qpos[dd["q"]] = math.radians(spec.open_deg) * rng.uniform(0.75, 0.98) if rng.uniform() < spec.open_prob else 0.0
        closet_open = d.qpos[self.doors["laundry_door"]["q"]] > 1.0
        for n, (q, v) in self.mdoor.items():       # machine doors swing out through the closet doorway
            d.qpos[q] = rng.uniform(0.5, 1.8) if closet_open and rng.uniform() < 0.3 else 0.0

    # ------------------------------------------------------------------- step
    def _enabled(self):
        """Motors that respond: not overheated, not broken, battery alive."""
        en = ~self.motors.tripped
        if self.broken[self.part_names.index("wrist")]:
            en[self.wrist_duty] = False
        if self.broken[self.part_names.index("head")]:
            en[self.head_duty] = False
        return en & (not self.battery.dead)

    def _apply_supply(self):
        """Motor gains follow their supply voltage; everything stops when the BMS opens."""
        m = self.model
        vb = self.battery.v if not self.battery.dead else 0.0
        supply = np.where(self.on_battery, vb, 12.0 if not self.battery.dead else 0.0)
        frac = supply / self.vrated
        gain = self.nominal["gainprm"][self.duty_act]
        m.actuator_gainprm[self.duty_act, 0] = np.where(self.position_mode, gain, gain * frac * self.strength)
        lim = self.stall * np.minimum(1.0, frac) * self.torque_limit * np.where(self.position_mode, self.strength, 1.0)
        lim = lim * self._enabled()
        m.actuator_forcerange[self.duty_act, 0] = -lim
        m.actuator_forcerange[self.duty_act, 1] = lim
        axis_on = 0.0 if self.battery.dead else 1.0
        m.actuator_forcerange[self.axis_act] = self.nominal["axis_force"] * axis_on
        if self.broken[self.part_names.index("mast")]:
            m.actuator_forcerange[self.act["lift"]] = 0.0

    def _external_forces(self):
        """Lift counterbalance spring, door latch and catches, and accidental shoves."""
        m, d = self.model, self.data
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[self.lift_v] = self.counterbalance
        for n, (q, v) in self.mdoor.items():          # magnetic catch on the machine doors
            a = d.qpos[q]
            if a < 0.12:
                d.qfrc_applied[v] -= 6.0 * (0.12 - a) / 0.12
        for dd in self.doors.values():                 # magnetic catch on the laundry closet door
            if dd["spec"].latch == "catch" and d.qpos[dd["q"]] < 0.08:
                d.qfrc_applied[dd["v"]] -= 4.0 * (0.08 - d.qpos[dd["q"]]) / 0.08
        if self.push is not None and self.push_left > 0:
            force, point = self.push
            mujoco.mj_applyFT(m, d, force, np.zeros(3), point, self.head_body, d.qfrc_applied)

    def _latch(self):
        """Lever doors are locked while shut unless a lever is pressed far enough."""
        d = self.data
        for dd in self.doors.values():
            spec = dd["spec"]
            if spec.latch != "lever":
                continue
            closed = d.qpos[dd["q"]] < 0.03
            pressed = d.qpos[dd["lever"]] > math.radians(spec.lever_unlatch_deg)
            self.model.jnt_range[dd["jid"]] = [0.0, 0.03] if (closed and not pressed) else dd["range"]

    def _maybe_push(self):
        rng = self.np_random
        if self.push_left > 0:
            self.push_left -= self.dt
            return
        self.push = None
        if self.cfg.push_rate > 0 and rng.uniform() < self.cfg.push_rate / 60.0 * self.dt:
            a = rng.uniform(-math.pi, math.pi)
            f = rng.uniform(*self.cfg.push_force)
            point = self.data.xpos[self.head_body].copy()
            point[2] -= rng.uniform(0.0, 0.6)
            self.push = (np.array([f * math.cos(a), f * math.sin(a), 0.0]), point)
            self.push_left = 0.2

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.battery.dead:
            a = np.zeros_like(a)
        self._apply_supply()
        self._latch()
        self._maybe_push()
        d.ctrl[:] = a

        motor_w, docked_frac = 0.0, 0.0
        sub_dt = 1.0 / S.PHYSICS_HZ
        vb = max(self.battery.v, 1.0)
        supply = np.where(self.on_battery, vb, 12.0)
        stall_v = self.stall * supply / self.vrated
        wnl = np.array([mo.no_load_speed for mo in S.DUTY_MOTORS])
        cmd = a[self.duty_act]
        for _ in range(self.n_sub):
            self._external_forces()
            mujoco.mj_step(m, d)
            tau, omega = d.actuator_force[self.duty_act], d.actuator_velocity[self.duty_act]
            # in position mode the servo's own loop picks the duty; infer it from torque and speed
            duty = np.where(self.position_mode, np.clip(tau / np.maximum(stall_v, 1e-6) + omega / wnl, -1, 1), cmd)
            i = self.motors.step(tau, duty, omega, sub_dt)
            motor_w += float((i * supply / np.where(self.on_battery, 1.0, S.ELECTRONICS.converter_efficiency)).sum()) / self.n_sub
            self._impacts(sub_dt)
            docked_frac += self._docked() / self.n_sub

        axis_w = 0.0
        if not self.battery.dead:
            for k, ax in zip(self.axis_act, (S.LIFT, S.ARM)):
                axis_w += ax.moving_power if abs(a[k]) > 0.02 else ax.holding_power
        load_i = (motor_w + axis_w + S.ELECTRONICS.idle_power()) / vb
        charge_i = self.charger.current(docked_frac > 0.5, self.battery, self.dt)
        e = self.battery.step(0.0 if self.battery.dead else load_i, charge_i, self.dt)
        self.energy_used += max(e, 0.0)
        self.energy_charged += max(-e, 0.0)
        self.broken |= self.part_damage > self.part_limit
        self._integrate_odom()
        self.prev_action = a
        self.t += self.dt

        prev = self.info
        self.info = self._info(charging=charge_i > 0, load_i=load_i, charge_i=charge_i)
        terminated = bool(self.info["dead"])
        truncated = self.t >= self.cfg.max_seconds
        reward = float(self.reward_fn(self.info, prev))
        return self._obs(), reward, terminated, truncated, dict(self.info)

    def _integrate_odom(self):
        d = self.data
        wl, wr = d.qvel[self.wheel_v]
        v = S.GEOM.wheel_radius * (wl + wr) / 2 * (1 + self.odom_bias[0])
        gz = d.sensordata[self.sensor_slice["imu_gyro"]][2] * (1 + self.odom_bias[1])
        self.odom[2] += gz * self.dt
        self.odom[0] += v * math.cos(self.odom[2]) * self.dt
        self.odom[1] += v * math.sin(self.odom[2]) * self.dt

    def _impacts(self, dt):
        """Hard contacts between fragile parts and anything not on the robot add to that part's damage."""
        d = self.data
        if d.ncon == 0:
            return
        g = d.contact.geom
        efc = d.contact.efc_address
        r1, r2 = self.robot_geom[g[:, 0]], self.robot_geom[g[:, 1]]
        part = np.where(r1, self.part_of[g[:, 0]], self.part_of[g[:, 1]])
        mask = (r1 ^ r2) & (part >= 0) & (efc >= 0)
        if not mask.any():
            return
        fn = d.efc_force[efc[mask]]
        excess = np.maximum(fn - self.part_thresh[part[mask]], 0.0) * dt
        np.add.at(self.part_damage, part[mask], excess)

    def _docked(self) -> bool:
        d = self.data
        if d.ncon == 0:
            return False
        g = d.contact.geom
        lo, hi = np.minimum(g[:, 0], g[:, 1]), np.maximum(g[:, 0], g[:, 1])
        touch = lambda a, b: bool(np.any((lo == min(a, b)) & (hi == max(a, b))))
        L = [touch(self.pads[0], c) for c in self.contacts]
        R = [touch(self.pads[1], c) for c in self.contacts]
        return sum(L) == 1 and sum(R) == 1 and L != R

    # -------------------------------------------------------------- sensing
    def _lidar(self):
        m, d, L = self.model, self.data, S.LIDAR
        if self.broken[self.part_names.index("lidar")]:
            return np.zeros(L.rays)
        R = d.site_xmat[self.lidar_site].reshape(3, 3)
        dirs = np.ascontiguousarray((self.lidar_dirs @ R.T).ravel())
        dist = np.zeros(L.rays)
        gid = np.zeros(L.rays, np.int32)
        mujoco.mj_multiRay(m, d, d.site_xpos[self.lidar_site], dirs, self.ray_groups, 1, -1, gid, dist, None,
                           L.rays, L.range[1])
        dist = np.where((dist < 0) | (dist > L.range[1]), L.range[1], dist)
        dist = np.maximum(dist, L.range[0])
        if self.cfg.sensor_noise:
            rng = self.np_random
            dist = dist + rng.normal(0, L.noise, L.rays)
            dist = np.where(rng.uniform(size=L.rays) < L.dropout, L.range[1], dist)
        return dist

    def _cliffs(self):
        m, d = self.model, self.data
        out = np.zeros(4)
        gid = np.zeros(1, np.int32)
        for k, s in enumerate(self.cliff_sites):
            down = -d.site_xmat[s].reshape(3, 3)[:, 2].copy()
            r = mujoco.mj_ray(m, d, d.site_xpos[s], down, self.ray_groups, 1, -1, gid)
            out[k] = 0.2 if r < 0 else min(r, 0.2)
        return out

    def _ir(self):
        m, d = self.model, self.data
        bpos = d.site_xpos[self.beacon]
        bx = d.site_xmat[self.beacon].reshape(3, 3)[:, 0]
        out = np.zeros(4)
        gid = np.zeros(1, np.int32)
        for k, s in enumerate(self.ir_sites):
            p = d.site_xpos[s]
            v = bpos - p
            dist = float(np.linalg.norm(v))
            u = v / max(dist, 1e-6)
            c = float(d.site_xmat[s].reshape(3, 3)[:, 0] @ u)
            if c <= 0:
                continue
            hit = mujoco.mj_ray(m, d, p, u, self.ray_groups, 1, -1, gid)
            if 0 <= hit < dist - 0.02:
                continue
            emit = max(float(bx @ -u), 0.0) ** 0.5
            out[k] = S.DOCK.ir_beacon_power * emit * c ** 2 / max(dist * dist, 0.04)
        return out

    def _obs(self):
        d, rng, N = self.data, self.np_random, S.NOISE
        nz = (lambda sd, n: rng.normal(0, sd, n)) if self.cfg.sensor_noise else (lambda sd, n: np.zeros(n))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.sensordata[self.sensor_slice["imu_quat"]])
        grav = R.reshape(3, 3).T @ np.array([0, 0, -1.0]) + nz(N.grav, 3)
        b = self.battery
        charging = float(self.info.get("charging", 0.0)) if hasattr(self, "info") else 0.0
        arm_ext = float(d.qpos[self.arm_q].sum())
        vec = np.concatenate([
            (d.qvel[self.wheel_v] + nz(N.vel, 2)) / 20.0,
            [self.odom[0] / 3, self.odom[1] / 3, math.sin(self.odom[2]), math.cos(self.odom[2])],
            [(d.qpos[self.lift_q] + nz(N.axis_pos, 1)[0]) / S.LIFT.stroke, d.qvel[self.lift_v] / S.LIFT.max_speed,
             (arm_ext + nz(N.axis_pos, 1)[0]) / S.ARM.stroke, float(d.qvel[self.arm_v].sum()) / S.ARM.max_speed],
            d.qpos[self.servo_q] + nz(N.enc, len(self.servo_q)),
            (d.qvel[self.servo_v] + nz(N.vel, len(self.servo_v))) / 5.0,
            d.actuator_force[self.duty_act] / self.stall, (self.motors.temp - 25.0) / 50.0,
            grav, (d.sensordata[self.sensor_slice["imu_gyro"]] + nz(N.gyro, 3)) / 5.0,
            d.sensordata[self.sensor_slice["imu_acc"]] / 9.81 + nz(N.acc / 9.81, 3),
            self._lidar() / S.LIDAR.range[1],
            self._cliffs() / 0.2, [min(d.sensordata[self.sensor_slice["bumper"]][0] / 50.0, 1.0)],
            [(b.v - S.BATTERY.nominal_v) / 5.0 + nz(N.v, 1)[0], b.i / 10.0 + nz(N.i, 1)[0], b.soc, charging],
            np.log1p(self._ir() * 10.0),
            (~self.broken).astype(float),
            self.prev_action,
        ]).astype(np.float32)
        if self.cfg.cameras:
            return {"state": vec, **self._camera_images()}
        return vec

    def _camera_images(self):
        h, w = self.cfg.cameras
        if not self._cam:
            self._cam = {"rgb": mujoco.Renderer(self.model, h, w)}
            if self.cfg.depth:
                self._cam["depth"] = mujoco.Renderer(self.model, h, w)
                self._cam["depth"].enable_depth_rendering()
        out = {}
        shots = [("head_rgb", "head_cam", "rgb", "head"), ("palm_rgb", "palm_cam", "rgb", "palm_cam")]
        if self.cfg.depth:
            shots.append(("head_depth", "head_cam", "depth", "head"))
        for key, cam, kind, part in shots:
            r = self._cam[kind]
            self._sync_hfield(r)
            r.update_scene(self.data, camera=cam)
            img = r.render().copy()
            if self.broken[self.part_names.index(part)]:
                img[:] = 0
            out[key] = img
        if self.cfg.depth:
            dep = out["head_depth"].astype(np.float32)
            lo, hi = S.DEPTH_UPGRADE.depth_range
            if self.cfg.sensor_noise:
                dep = dep * (1 + self.np_random.normal(0, S.DEPTH_UPGRADE.depth_noise, dep.shape))
            out["head_depth"] = np.where((dep < lo) | (dep > hi), 0.0, dep).astype(np.float32)
        return out

    def _sync_hfield(self, r):
        if getattr(r, "_hf_version", -1) != self.terrain.version:
            r._gl_context.make_current()
            for h in range(self.model.nhfield):
                mujoco.mjr_uploadHField(self.model, r._mjr_context, h)
            r._hf_version = self.terrain.version

    def _info(self, charging=False, load_i=0.0, charge_i=0.0):
        d = self.data
        pos = d.xpos[self.robot]
        Rm = d.xmat[self.robot].reshape(3, 3)
        up = float(Rm[2, 2])
        tipped = up < 0.6
        fatal = any(self.broken[self.part_names.index(p)] for p in TERMINAL_PARTS)
        tidy, where = self.items.tidy_count(d)
        return {
            "t": self.t, "soc": self.battery.soc, "v": self.battery.v, "load_a": load_i, "charge_a": charge_i,
            "power_w": load_i * self.battery.v, "charging": bool(charging), "battery_dead": bool(self.battery.dead),
            "tipped": bool(tipped), "broken_parts": [p for p, b in zip(self.part_names, self.broken) if b],
            "part_damage": dict(zip(self.part_names, np.round(self.part_damage, 2).tolist())),
            "dead": bool(self.battery.dead or tipped or fatal),
            "pos": pos.copy(), "yaw": float(math.atan2(Rm[1, 0], Rm[0, 0])), "upright": up,
            "room": H.room_of(pos[:2]),
            "lift_height": float(S.GEOM.lift_low + 0.02 + d.qpos[self.lift_q]),
            "arm_extension": float(d.qpos[self.arm_q].sum()),
            "dock_dist": float(np.linalg.norm((d.site_xpos[self.dock_target] - pos)[:2])),
            "doors": {n: round(float(d.qpos[dd["q"]]), 3) for n, dd in self.doors.items()},
            "door_latched": {n: bool(self.model.jnt_range[dd["jid"]][1] < 0.1) for n, dd in self.doors.items()},
            "washer_door": float(d.qpos[self.mdoor["washer"][0]]), "dryer_door": float(d.qpos[self.mdoor["dryer"][0]]),
            "tidy": tidy, "items": where, "items_active": int(self.items.active.sum()),
            "motor_temp_max": float(self.motors.temp.max()), "pushed": self.push is not None,
            "energy_used_wh": self.energy_used / 3600.0, "energy_charged_wh": self.energy_charged / 3600.0,
        }

    # ------------------------------------------------------------- render
    def render(self, camera="overview", width=960, height=544):
        if self._renderer is None or self._renderer.width != width:
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._sync_hfield(self._renderer)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        for r in [self._renderer, *self._cam.values()]:
            if r is not None:
                r.close()
