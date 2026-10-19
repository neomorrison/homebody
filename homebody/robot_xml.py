"""Generate the robot's MJCF from ``spec``.

    base (hub wheels, casters, outriggers, ballast, deck tote, lidar, bumper,
          cliff sensors, rear charge pads, dock beacon receivers)
      mast -> head_pan -> head_tilt -> head camera
      mast -> carriage (lift) -> stage 1 -> stage 2 -> stage 3 (telescope, out to the right)
            -> wrist_yaw -> wrist_pitch -> wrist_roll -> palm (camera) -> two fingers

Duty-driven motors use a DC-motor model whose gain tracks their supply voltage
(``env`` updates it). The lift and telescope take speed commands, like the
stepper drivers on the real robot. The three telescope stages are coupled by
equality constraints, as the belt couples them on the real arm.

Collision bits: world geoms are contype=1/conaffinity=1. Base, mast and head
are 2/1 (hit the world, not each other). Arm, wrist and gripper are 4/3: they
hit the world and the base, but not each other.
"""
from __future__ import annotations

import math

from . import spec as S

G = S.GEOM
Z0 = (G.base_bottom + G.base_top) / 2          # root height above the floor


def _f(*v):
    return " ".join(f"{x:.5g}" for x in v)


def defaults_xml() -> str:
    return """
  <default>
    <default class="base">
      <geom contype="2" conaffinity="1" condim="3" friction="0.8 0.02 0.001" group="2"/>
      <joint limited="true"/>
    </default>
    <default class="arm">
      <geom contype="4" conaffinity="3" condim="3" friction="0.8 0.02 0.001" group="2"/>
      <joint limited="true"/>
    </default>
    <default class="finger">
      <geom contype="4" conaffinity="3" condim="4" friction="1.2 0.02 0.001" priority="3" group="2"
            solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </default>
    <default class="glide">
      <geom contype="2" conaffinity="1" condim="1" friction="0 0 0" priority="3" group="2"/>
    </default>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="2"/>
    </default>
  </default>"""


def materials_xml() -> str:
    return """
    <material name="frame" rgba="0.92 0.92 0.90 1"/>
    <material name="ply" rgba="0.80 0.68 0.50 1"/>
    <material name="servo" rgba="0.12 0.12 0.14 1"/>
    <material name="rubber" rgba="0.10 0.10 0.10 1"/>
    <material name="alu" rgba="0.70 0.72 0.75 1"/>
    <material name="steel" rgba="0.35 0.35 0.38 1"/>
    <material name="pad" rgba="0.85 0.50 0.25 1"/>
    <material name="battery" rgba="0.15 0.35 0.75 1"/>
    <material name="pcb" rgba="0.10 0.50 0.25 1"/>
    <material name="lens" rgba="0.05 0.05 0.05 1"/>
    <material name="tpu" rgba="0.95 0.55 0.15 1"/>
    <material name="tote" rgba="0.25 0.55 0.75 0.85"/>"""


def _servo(name, motor, pos, euler=(0, 0, 0), cls="arm"):
    hs = (0.0225, 0.012, 0.0185) if motor is S.STS3250 else (0.0225, 0.012, 0.0165)
    return (f'<geom name="servo_{name}" type="box" size="{_f(*hs)}" pos="{_f(*pos)}" '
            f'euler="{_f(*(math.degrees(e) for e in euler))}" mass="{motor.mass}" material="servo" class="{cls}"/>')


def _hinge(j: S.Joint, cls="arm"):
    lo, hi = j.range
    m = j.motor
    if j.slide:     # rack and pinion: friction reflects through the gear; inertia and damping are kept
        g = j.gear  # modest (the full reflected rotor inertia, ~30 kg, makes the jaws bounce off objects)
        return (f'<joint name="{j.name}" type="slide" axis="{_f(*j.axis)}" range="{lo} {hi}" class="{cls}" '
                f'armature="1.0" damping="5" frictionloss="{m.friction * g:.4g}"/>')
    return (f'<joint name="{j.name}" axis="{_f(*j.axis)}" range="{lo} {hi}" class="{cls}" '
            f'armature="{m.armature}" damping="{m.damping}" frictionloss="{m.friction}"/>')


J = {j.name: j for j in S.SERVOS}


def _gripper_xml() -> str:
    """Parallel jaws: both fingers slide apart symmetrically (the right one follows the left through an
    equality constraint, as one pinion drives two racks)."""
    m = S.MASS
    fl = G.finger_length
    g = J["gripper"]
    lo, hi = g.range
    fingers = []
    for side, s in (("l", 1), ("r", -1)):
        joint = (_hinge(g) if side == "l" else
                 f'<joint name="gripper_r" type="slide" axis="0 -1 0" range="{lo} {hi}" class="arm" armature="0.5" damping="0.5"/>')
        fingers.append(f"""
                    <body name="finger_{side}" pos="0 0 -0.012">
                      {joint}
                      <geom name="finger_{side}" type="box" size="0.015 0.006 {fl / 2:.4f}" pos="0 {s * 0.008:.3f} {-fl / 2:.4f}"
                            mass="{m['finger']}" material="tpu" class="finger"/>
                    </body>""")
    return f"""
                  <body name="palm" pos="0 0 -0.035">
                    <geom name="palm" type="box" size="0.03 0.075 0.012" mass="{m['palm']}" material="frame" class="arm"/>
                    {_servo("gripper", S.STS3250, (0.0, 0, 0.02), (0, 0, 0))}
                    <camera name="palm_cam" pos="0.04 0 -0.005" xyaxes="0 1 0 -0.94 0 0.34" fovy="{S.PALM_CAM.fovy}"/>
                    <site name="grip_center" pos="0 0 {-0.012 - fl * 0.75:.4f}" size="0.006" rgba="0 1 1 0.4"/>
                    {''.join(fingers)}
                  </body>"""


def _arm_xml() -> str:
    m = S.MASS
    L, n = G.stage_length, G.n_stages
    stroke = S.ARM.stroke / n
    stages_open, stages_close = [], []
    for k in range(n):
        w = 0.022 - 0.004 * k
        pos = f"0 {G.arm_root_y - G.mast_y - L / 2:.4f} 0" if k == 0 else "0 0 0"
        stages_open.append(f"""
          <body name="stage_{k + 1}" pos="{pos}">
            <joint name="arm_{k + 1}" type="slide" axis="0 -1 0" range="0 {stroke:.4f}" class="arm"
                   frictionloss="{S.ARM.friction / n:.3g}" damping="2" armature="{1.2 if k == 0 else 0.1}"/>
            <geom name="stage_{k + 1}" type="box" size="{_f(w, L / 2, w)}" mass="{m['stage']}" material="alu" class="arm"/>""")
        stages_close.append("          </body>")
    wrist = f"""
            <body name="wrist_yaw" pos="0 {-L / 2 + 0.01:.4f} -0.02">
              {_hinge(J["wrist_yaw"])}
              {_servo("wrist_yaw", S.STS3215, (0, 0, 0), (0, 0, 0))}
              <geom type="box" size="0.012 0.02 {G.wrist_drop / 2:.4f}" pos="0 0 {-G.wrist_drop / 2:.4f}" mass="{m['wrist_bracket']}" material="frame" class="arm"/>
              <body name="wrist_pitch" pos="0 0 {-G.wrist_drop:.4f}">
                {_hinge(J["wrist_pitch"])}
                {_servo("wrist_pitch", S.STS3250, (0, 0, 0), (0, math.pi / 2, 0))}
                <body name="wrist_roll" pos="0 0 -0.03">
                  {_hinge(J["wrist_roll"])}
                  {_servo("wrist_roll", S.STS3215, (0, 0, -0.005), (0, 0, 0))}
                  {_gripper_xml()}
                </body>
              </body>
            </body>"""
    return "".join(stages_open) + wrist + "\n".join(stages_close)


def body_xml(name="robot") -> str:
    m = S.MASS
    hx, hy = G.base_length / 2, G.base_width / 2
    hz = (G.base_top - G.base_bottom) / 2
    top = G.base_top - Z0
    wz = G.wheel_radius - Z0
    wheels = "".join(f"""
    <body name="{w}" pos="0 {s * G.wheel_track / 2:.4f} {wz:.4f}">
      <joint name="{w}" axis="0 1 0" limited="false" armature="{S.HUB_MOTOR.armature}" damping="{S.HUB_MOTOR.damping}" frictionloss="{S.HUB_MOTOR.friction}"/>
      <geom name="{w}" type="cylinder" size="{G.wheel_radius} {G.wheel_width / 2}" euler="90 0 0" mass="{S.HUB_MOTOR.mass}" material="rubber"/>
    </body>""" for w, s in (("wheel_l", 1), ("wheel_r", -1)))
    casters = "".join(
        f'<geom name="caster_{k}" type="sphere" size="{G.caster_radius}" pos="{_f(x, 0, G.caster_radius - Z0)}" '
        f'mass="0.08" material="alu" class="glide"/>'
        for k, x in (("front", G.caster_x), ("rear", -G.caster_x)))
    ox, oy = G.outrigger_xy
    orz = G.outrigger_radius + G.outrigger_clearance - Z0
    outriggers = "".join(
        f'<geom name="outrigger_{k}" type="sphere" size="{G.outrigger_radius}" pos="{_f(sx * ox, sy * oy, orz)}" '
        f'mass="0.05" material="rubber" class="glide"/>'
        f'<geom type="box" size="0.03 0.012 0.008" pos="{_f(sx * (ox - 0.03), sy * oy, orz + 0.02)}" mass="0.03" material="steel" class="visual"/>'
        for k, (sx, sy) in (("fl", (1, 1)), ("fr", (1, -1)), ("rl", (-1, 1)), ("rr", (-1, -1))))
    cliffs = "".join(
        f'<site name="cliff_{k}" pos="{_f(sx * (hx - 0.02), sy * (hy - 0.02), -hz - 0.002)}" size="0.004" rgba="1 0 0 1"/>'
        for k, (sx, sy) in (("fl", (1, 1)), ("fr", (1, -1)), ("rl", (-1, 1)), ("rr", (-1, -1))))
    ir = "".join(
        f'<site name="ir_{k}" pos="{_f(px, py, top - 0.01)}" euler="0 0 {a}" size="0.005" rgba=".3 .1 .1 1"/>'
        for k, px, py, a in (("front", hx, 0, 0), ("left", 0, hy, 90), ("back", -hx, 0, 180), ("right", 0, -hy, -90)))
    pads = "".join(
        f'<geom name="pad_{k}" type="box" size="0.002 0.02 0.012" pos="{_f(-hx - 0.002, s * G.dock_pad_y, G.dock_pad_z - Z0)}" '
        f'mass="0.005" material="pad"/>' for k, s in (("l", 1), ("r", -1)))
    tx0, tx1 = G.tote_x
    ty0, ty1 = G.tote_y
    th, t = G.tote_height, 0.004
    tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
    tote = "".join([
        f'<geom name="tote_px" type="box" size="{_f(t, (ty1 - ty0) / 2 + t, th / 2)}" pos="{_f(tx1 + t, tcy, top + th / 2)}" mass="{m["tote"] / 4}" material="tote"/>',
        f'<geom name="tote_nx" type="box" size="{_f(t, (ty1 - ty0) / 2 + t, th / 2)}" pos="{_f(tx0 - t, tcy, top + th / 2)}" mass="{m["tote"] / 4}" material="tote"/>',
        f'<geom name="tote_py" type="box" size="{_f((tx1 - tx0) / 2, t, th / 2)}" pos="{_f(tcx, ty1 + t, top + th / 2)}" mass="{m["tote"] / 4}" material="tote"/>',
        f'<geom name="tote_ny" type="box" size="{_f((tx1 - tx0) / 2, t, th / 2)}" pos="{_f(tcx, ty0 - t, top + th / 2)}" mass="{m["tote"] / 4}" material="tote"/>',
        f'<site name="tote" type="box" size="{_f((tx1 - tx0) / 2, (ty1 - ty0) / 2, th / 2)}" pos="{_f(tcx, tcy, top + th / 2)}" rgba="0 0 0 0"/>',
    ])
    mast_h = (G.mast_top - G.base_top) / 2
    head_pan, head_tilt = J["head_pan"], J["head_tilt"]
    carriage_z = G.lift_low - Z0 + 0.02
    return f"""
  <body name="{name}" pos="0 0 {Z0 + 0.002:.4f}" childclass="base">
    <freejoint name="root"/>
    <site name="imu" pos="0 0 0" size="0.008" rgba="0 1 0 1"/>
    <geom name="chassis" type="box" size="{_f(hx, hy, hz)}" mass="{m['chassis']}" material="ply"/>
    <geom name="ballast" type="box" size="0.18 0.12 0.012" pos="0 0 {-hz + 0.014:.4f}" mass="{m['ballast']}" material="steel" class="visual"/>
    <geom name="battery" type="box" size="0.09 0.04 0.035" pos="-0.12 0 {-hz + 0.04:.4f}" mass="{S.BATTERY.mass}" material="battery" class="visual"/>
    <geom name="electronics" type="box" size="0.05 0.04 0.01" pos="0.15 0 {hz - 0.012:.4f}" mass="{m['electronics']}" material="pcb" class="visual"/>
    <geom name="bumper" type="box" size="0.012 {hy + 0.03:.4f} 0.03" pos="{hx + 0.012:.4f} 0 -0.005" mass="0.05" material="rubber"/>
    <site name="bumper" type="box" size="0.02 {hy + 0.03:.4f} 0.04" pos="{hx + 0.014:.4f} 0 -0.005" rgba="0 0 0 0"/>
    <geom name="lidar" type="cylinder" size="0.035 0.02" pos="{_f(G.lidar_x, 0, G.lidar_z - Z0 - 0.005)}" mass="{m['lidar']}" material="servo"/>
    <site name="lidar" pos="{_f(G.lidar_x, 0, G.lidar_z - Z0)}" size="0.006" rgba="1 0 0 1"/>
    {tote}
    <geom name="mast" type="box" size="0.02 0.04 {mast_h:.4f}" pos="{_f(G.mast_x, G.mast_y, top + mast_h)}" mass="{m['mast']}" material="alu"/>
    <geom name="mast_cap" type="box" size="0.035 0.05 0.02" pos="{_f(G.mast_x, G.mast_y, top + 2 * mast_h + 0.02)}" mass="0.05" material="rubber"/>
    <body name="head" pos="{_f(G.mast_x, G.mast_y, G.mast_top - Z0 + 0.06)}">
      {_hinge(head_pan, "base")}
      {_servo("head_pan", S.STS3215, (0, 0, -0.02), (0, 0, 0), "base")}
      <body name="head_tilt" pos="0 0 0.02">
        {_hinge(head_tilt, "base")}
        <geom name="head" type="box" size="0.025 0.05 0.02" mass="{m['head']}" material="servo"/>
        <geom name="head_lens" type="cylinder" size="0.01 0.003" pos="0.026 0 0" euler="0 90 0" material="lens" class="visual"/>
        <camera name="head_cam" pos="0.03 0 0" xyaxes="0 -1 0 0 0 1" fovy="{S.HEAD_CAM.fovy}"/>
      </body>
    </body>
    {pads}
    {cliffs}
    {ir}
    {wheels}
    {casters}
    {outriggers}
    <body name="carriage" pos="{_f(G.arm_x, G.mast_y, carriage_z)}" childclass="arm">
      <joint name="lift" type="slide" axis="0 0 1" range="0 {S.LIFT.stroke:.4f}" frictionloss="{S.LIFT.friction}" damping="10" armature="0.5"/>
      <geom name="carriage" type="box" size="0.022 0.05 0.04" mass="{m['carriage']}" material="alu"/>
      {_arm_xml()}
    </body>
  </body>"""


def servo_position_params(j: S.Joint):
    """(gain, bias0, bias1, bias2) so that force = kp * (target - q) - kd * qdot with target = mid + a * half."""
    lo, hi = j.range
    if not j.slide:
        lo, hi = math.radians(lo), math.radians(hi)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    kp = j.motor.stall_torque / j.stiffness_err               # motor torque per unit joint error
    kd = kp * 0.04
    # MuJoCo's actuator length and velocity are in motor units (gear x joint), hence the / gear
    return kp * half, kp * mid, -kp / j.gear, -kd / j.gear


def actuators_xml(servo_mode: str = S.SERVO_MODE) -> str:
    """Actuators in ``spec.ACTION_NAMES`` order (gains at rated voltage; env rescales)."""
    out = []
    for name in S.ACTION_NAMES:
        if name in ("lift", "arm"):
            ax = S.LIFT if name == "lift" else S.ARM
            joint, ratio = ("lift", 1) if name == "lift" else ("arm_1", G.n_stages)
            kv = 4000.0 if name == "lift" else 1500.0     # speed-loop stiffness; stable at 250 Hz
            out.append(
                f'<general name="m_{name}" joint="{joint}" ctrlrange="-1 1" gainprm="{kv * ax.max_speed / ratio:.5g}" '
                f'biastype="affine" biasprm="0 0 {-kv:.5g}" forcerange="{-ax.max_force * ratio} {ax.max_force * ratio}"/>')
            continue
        if name in J and servo_mode == "position":
            j = J[name]
            g0, b0, b1, b2 = servo_position_params(j)
            lim = j.motor.stall_torque * j.torque_limit
            out.append(
                f'<general name="m_{name}" joint="{name}" gear="{j.gear:g}" ctrlrange="-1 1" gainprm="{g0:.5g}" biastype="affine" '
                f'biasprm="{b0:.5g} {b1:.5g} {b2:.5g}" forcerange="{-lim:.5g} {lim:.5g}"/>')
            continue
        mo = S.HUB_MOTOR if name in S.WHEELS else J[name].motor
        gear = J[name].gear if name in J else 1.0
        out.append(
            f'<general name="m_{name}" joint="{name}" gear="{gear:g}" ctrlrange="-1 1" gainprm="{mo.stall_torque:.5g}" biastype="affine" '
            f'biasprm="0 0 {-mo.stall_torque / mo.no_load_speed:.5g}" '
            f'forcerange="{-mo.stall_torque:.5g} {mo.stall_torque:.5g}"/>')
    return "\n    ".join(out)


def equality_xml() -> str:
    eq = [f'<joint joint1="arm_{k}" joint2="arm_1" polycoef="0 1 0 0 0" solref="0.004 1"/>'
          for k in range(2, G.n_stages + 1)]
    eq.append('<joint joint1="gripper_r" joint2="gripper" polycoef="0 1 0 0 0" solref="0.004 1"/>')
    return "\n    ".join(eq)


def sensors_xml() -> str:
    return """
    <accelerometer name="imu_acc" site="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <touch name="bumper" site="bumper"/>"""
