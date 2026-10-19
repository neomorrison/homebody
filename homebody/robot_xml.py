"""Generate the robot's MJCF from ``spec``.

Each servo is a real box with its real mass where it sits on the frame, each
3D-printed link carries its bracket mass, and each actuator is a DC-motor model
(see ``spec.ServoSpec``) driven by a PWM duty cycle. Battery voltage enters as
the actuator gain, which ``env`` updates every control step.
"""
from __future__ import annotations

import math

from . import spec as S

G = S.GEOM

# servo case half-sizes (m)
SERVO_BOX = {
    S.STS3250.name: (0.0225, 0.012, 0.0185),
    S.STS3215_12V.name: (0.0225, 0.012, 0.0165),
}
JOINT_BY_NAME = {j.name: j for j in S.JOINTS}


def _f(*v):
    return " ".join(f"{x:.5g}" for x in v)


def _servo_geom(joint_name, pos, euler=(0, 0, 0)):
    """Servo case as a visual + mass box. It also collides (cases are what hit the floor)."""
    j = JOINT_BY_NAME[joint_name]
    hs = SERVO_BOX[j.servo.name]
    return (f'<geom name="servo_{joint_name}" type="box" size="{_f(*hs)}" pos="{_f(*pos)}" '
            f'euler="{_f(*(math.degrees(e) for e in euler))}" mass="{j.servo.mass}" material="servo" class="body"/>')


def _joint(name):
    j = JOINT_BY_NAME[name]
    lo, hi = j.range_deg
    return f'<joint name="{name}" axis="{_f(*j.axis)}" range="{lo} {hi}" class="servo_joint"/>'


def _leg(side):
    s = 1 if side == "l" else -1
    hy = s * G.hip_half_width
    m = S.FRAME_MASS
    foot_cx = G.foot_length / 2 - G.foot_heel
    sole_t = 0.008
    fx, fy = G.foot_length / 2 - 0.012, G.foot_width / 2 - 0.010
    fsr = "".join(
        f'<site name="fsr_{side}_{k}" type="box" size="0.010 0.010 0.004" '
        f'pos="{_f(foot_cx + dx * fx, dy * fy, -G.ankle_to_sole + 0.002)}" rgba="1 .8 .2 .6"/>'
        for k, (dx, dy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)])
    )
    return f"""
    <body name="hip_yaw_{side}" pos="0 {hy:.4f} {-G.hip_drop:.4f}">
      {_joint(f"hip_yaw_{side}")}
      <geom type="box" size="0.02 0.025 0.008" pos="0 0 -0.008" mass="{m['hip_yaw_link']}" material="frame" class="visual"/>
      {_servo_geom(f"hip_roll_{side}", (0, 0, -0.012), (0, 0, 0))}
      <body name="hip_roll_{side}">
        {_joint(f"hip_roll_{side}")}
        <geom type="box" size="0.025 0.018 0.01" pos="0 0 -0.01" mass="{m['hip_roll_link']}" material="frame" class="visual"/>
        {_servo_geom(f"hip_pitch_{side}", (0, s * 0.0, -0.02), (math.pi / 2, 0, 0))}
        <body name="thigh_{side}">
          {_joint(f"hip_pitch_{side}")}
          <geom type="capsule" fromto="0 0 -0.02 0 0 {-G.thigh + 0.02:.4f}" size="0.022" mass="{m['thigh']}" material="frame" class="body"/>
          {_servo_geom(f"knee_{side}", (0, 0, -G.thigh + 0.035), (math.pi / 2, 0, 0))}
          <body name="shank_{side}" pos="0 0 {-G.thigh:.4f}">
            {_joint(f"knee_{side}")}
            <geom type="capsule" fromto="0 0 -0.02 0 0 {-G.shank + 0.03:.4f}" size="0.02" mass="{m['shank']}" material="frame" class="body" contype="6" conaffinity="5"/>
            {_servo_geom(f"ankle_pitch_{side}", (0, 0, -G.shank + 0.04), (math.pi / 2, 0, 0))}
            <body name="ankle_{side}" pos="0 0 {-G.shank:.4f}">
              {_joint(f"ankle_pitch_{side}")}
              <geom type="box" size="0.012 0.02 0.006" mass="{m['ankle_link']}" material="frame" class="visual"/>
              <body name="foot_{side}">
                {_joint(f"ankle_roll_{side}")}
                {_servo_geom(f"ankle_roll_{side}", (0.0, 0, -0.012), (0, 0, 0))}
                <geom name="sole_{side}" type="box" size="{_f(G.foot_length / 2, G.foot_width / 2, sole_t / 2)}"
                      pos="{_f(foot_cx, 0, -G.ankle_to_sole + sole_t / 2)}" mass="{m['foot']}" material="sole" class="foot"/>
                {fsr}
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def _arm(side):
    s = 1 if side == "l" else -1
    m = S.FRAME_MASS
    return f"""
    <body name="shoulder_{side}" pos="0 {s * G.shoulder_half_width:.4f} {G.torso_height:.4f}">
      {_joint(f"shoulder_pitch_{side}")}
      <geom type="box" size="0.012 0.015 0.015" mass="{m['upper_arm_a']}" material="frame" class="visual"/>
      {_servo_geom(f"shoulder_roll_{side}", (0, s * 0.02, -0.01), (0, 0, math.pi / 2))}
      <body name="upper_arm_{side}" pos="0 {s * 0.02:.4f} 0">
        {_joint(f"shoulder_roll_{side}")}
        <geom type="capsule" fromto="0 0 -0.02 0 0 {-G.upper_arm + 0.02:.4f}" size="0.018" mass="{m['upper_arm']}" material="frame" class="body"/>
        {_servo_geom(f"elbow_{side}", (0, 0, -G.upper_arm + 0.03), (math.pi / 2, 0, 0))}
        <body name="forearm_{side}" pos="0 0 {-G.upper_arm:.4f}">
          {_joint(f"elbow_{side}")}
          <geom type="capsule" fromto="0 0 -0.01 0 0 {-G.forearm:.4f}" size="0.016" mass="{m['forearm']}" material="frame" class="body"/>
          <geom name="hand_{side}" type="sphere" size="0.022" pos="0 0 {-G.forearm:.4f}" mass="0.01" material="frame" class="body" contype="6" conaffinity="5"/>
        </body>
      </body>
    </body>"""


def defaults_xml() -> str:
    return """
  <default>
    <!-- Collision bits: world geoms are contype=1/conaffinity=1. Robot parts are
         contype=2/conaffinity=1, so they hit the world but not each other (the hip
         and shoulder clusters are designed not to self-collide). Shanks, feet and
         hands add bit 4 on both sides so legs cannot pass through legs. -->
    <default class="servo_joint">
      <joint limited="true" damping="0.02" armature="0.01" frictionloss="0.05"/>
    </default>
    <default class="body">
      <geom contype="2" conaffinity="1" condim="3" friction="0.8 0.02 0.001" group="2"/>
    </default>
    <default class="foot">
      <geom contype="6" conaffinity="5" condim="4" friction="0.9 0.02 0.001" priority="1" group="2"/>
    </default>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="2"/>
    </default>
  </default>"""


def materials_xml() -> str:
    return """
    <material name="frame" rgba="0.92 0.92 0.90 1"/>
    <material name="servo" rgba="0.12 0.12 0.14 1"/>
    <material name="sole" rgba="0.80 0.45 0.20 1"/>
    <material name="battery" rgba="0.15 0.35 0.75 1"/>
    <material name="pcb" rgba="0.10 0.50 0.25 1"/>
    <material name="lens" rgba="0.05 0.05 0.05 1"/>"""


def body_xml(name="robot") -> str:
    """The robot's worldbody subtree. Root is the torso (pelvis and chest are one rigid frame)."""
    m = S.FRAME_MASS
    h0 = G.standing_pelvis_height + 0.01
    neck_z = G.torso_height + 0.025
    head_z = G.neck + G.head_radius
    ir = "".join(
        f'<site name="ir_{k}" pos="{_f(G.head_radius * math.cos(a), G.head_radius * math.sin(a), 0)}" '
        f'euler="0 0 {math.degrees(a):.4f}" size="0.006" rgba=".3 .1 .1 1"/>'
        for k, a in zip(("front", "left", "back", "right"), (0, math.pi / 2, math.pi, -math.pi / 2))
    )
    return f"""
  <body name="{name}" pos="0 0 {h0:.4f}" childclass="body">
    <freejoint name="root"/>
    <site name="imu" pos="0 0 0.06" size="0.008" rgba="0 1 0 1"/>
    <geom name="pelvis" type="box" size="0.035 {G.hip_half_width + 0.01:.4f} 0.022" pos="0 0 -0.01" mass="{m['pelvis']}" material="frame"/>
    <geom name="chest" type="box" size="{_f(G.torso_depth / 2, G.torso_width / 2, (G.torso_height - 0.03) / 2)}"
          pos="0 0 {(G.torso_height + 0.03) / 2 - 0.01:.4f}" mass="{m['torso']}" material="frame"/>
    <geom name="battery" type="box" size="0.035 0.028 0.024" pos="-0.005 0 0.05" mass="{S.BATTERY.mass}" material="battery" class="visual"/>
    <geom name="computer" type="box" size="0.03 0.045 0.008" pos="0.028 0 0.11" mass="0.075" material="pcb" class="visual"/>
    <geom name="electronics" type="box" size="0.02 0.03 0.006" pos="0.028 0 0.05" mass="0.06" material="pcb" class="visual"/>
    {_servo_geom("hip_yaw_l", (0, G.hip_half_width, -0.012), (0, 0, 0))}
    {_servo_geom("hip_yaw_r", (0, -G.hip_half_width, -0.012), (0, 0, 0))}
    {_servo_geom("shoulder_pitch_l", (0, G.shoulder_half_width - 0.03, G.torso_height - 0.01), (0, 0, math.pi / 2))}
    {_servo_geom("shoulder_pitch_r", (0, -G.shoulder_half_width + 0.03, G.torso_height - 0.01), (0, 0, math.pi / 2))}
    {_servo_geom("neck_yaw", (0, 0, G.torso_height - 0.005), (0, 0, 0))}
    {_leg("l")}
    {_leg("r")}
    {_arm("l")}
    {_arm("r")}
    <body name="neck" pos="0 0 {neck_z:.4f}">
      {_joint("neck_yaw")}
      {_servo_geom("neck_pitch", (0, 0, 0.015), (math.pi / 2, 0, 0))}
      <body name="head" pos="0 0 0.015">
        {_joint("neck_pitch")}
        <geom name="head" type="sphere" size="{G.head_radius}" pos="0 0 {head_z - 0.015:.4f}" mass="{m['head']}" material="frame"/>
        <geom name="lens" type="cylinder" size="0.012 0.004" pos="{G.head_radius - 0.002:.4f} 0 {head_z - 0.005:.4f}" euler="0 90 0" material="lens" class="visual"/>
        <camera name="eye" pos="{G.head_radius:.4f} 0 {head_z - 0.005:.4f}" xyaxes="0 -1 0 0 0 1" fovy="100"/>
        <body name="ir_ring" pos="0 0 {head_z - 0.015:.4f}">
          {ir}
        </body>
      </body>
    </body>
  </body>"""


def actuators_xml(v_battery: float = 12.0) -> str:
    """One DC motor per joint: force = gain*duty + bias*omega, gain = stall*V/V_rated."""
    out = []
    for j in S.JOINTS:
        sv = j.servo
        gain = sv.stall_torque * v_battery / sv.v_rated
        back_emf = sv.stall_torque / sv.no_load_speed
        out.append(
            f'<general name="m_{j.name}" joint="{j.name}" ctrlrange="-1 1" ctrllimited="true" '
            f'gainprm="{gain:.5g}" biastype="affine" biasprm="0 0 {-back_emf:.5g}" '
            f'forcerange="{-sv.stall_torque:.5g} {sv.stall_torque:.5g}" forcelimited="true"/>'
        )
    return "\n    ".join(out)


def sensors_xml() -> str:
    fsr = "".join(f'<touch name="fsr_{s}_{k}" site="fsr_{s}_{k}"/>' for s in "lr" for k in range(4))
    return f"""
    <accelerometer name="imu_acc" site="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    {fsr}"""


def joint_param_overrides(model):
    """Set per-joint armature / damping / friction from each joint's servo."""
    import mujoco
    for j in S.JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.name)
        dof = model.jnt_dofadr[jid]
        model.dof_armature[dof] = j.servo.armature
        model.dof_damping[dof] = j.servo.damping
        model.dof_frictionloss[dof] = j.servo.friction
