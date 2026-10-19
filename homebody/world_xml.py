"""MJCF for the home (``home``), the robot (``robot_xml``), the floor (``terrain``) and items.

Every world geom has contact priority 2 and the robot's wheels and body 0, so
friction and softness of a contact are the surface's (see ``terrain``).
"""
from __future__ import annotations

import math

from . import home as H
from . import items as I
from . import robot_xml as R
from . import spec as S
from . import terrain as T

SURF = 'priority="2" condim="3"'


def _f(*v):
    return " ".join(f"{x:.5g}" for x in v)


def look_at(name, pos, target, fovy=60):
    """A fixed camera at ``pos`` aimed at ``target``."""
    import numpy as np
    p, t = np.array(pos, float), np.array(target, float)
    fwd = (t - p) / np.linalg.norm(t - p)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return f'<camera name="{name}" pos="{_f(*p)}" xyaxes="{_f(*right, *up)}" fovy="{fovy}"/>'


def walls_xml() -> str:
    X, Y = H.EXTENT[0] / 2, H.EXTENT[1] / 2
    t, h = H.WALL_T, H.WALL_H
    w = f'material="wall" {SURF} friction="0.5 0.02 0.001"'
    out = [   # outer walls sit outside the extent so the rooms keep their full size
        f'<geom name="wall_n" type="box" size="{_f(X + t, t / 2, h / 2)}" pos="{_f(0, Y + t / 2, h / 2)}" {w}/>',
        f'<geom name="wall_s" type="box" size="{_f(X + t, t / 2, h / 2)}" pos="{_f(0, -Y - t / 2, h / 2)}" {w}/>',
        f'<geom name="wall_e" type="box" size="{_f(t / 2, Y, h / 2)}" pos="{_f(X + t / 2, 0, h / 2)}" {w}/>',
        f'<geom name="wall_w" type="box" size="{_f(t / 2, Y, h / 2)}" pos="{_f(-X - t / 2, 0, h / 2)}" {w}/>',
    ]
    for k, (x0, y0, x1, y1) in enumerate(H.WALLS):
        # thickness across the wall only: extending along it would eat into doorways
        hx, hy = (abs(x1 - x0) / 2, t / 2) if abs(y1 - y0) < 1e-9 else (t / 2, abs(y1 - y0) / 2)
        out.append(f'<geom name="wall_{k}" type="box" size="{_f(hx, hy, h / 2)}" pos="{_f((x0 + x1) / 2, (y0 + y1) / 2, h / 2)}" {w}/>')
    return "\n    ".join(out)


def furniture_xml() -> str:
    return "\n    ".join(
        f'<body name="{f.name}" pos="{_f(*f.center, 0)}" euler="0 0 {math.degrees(f.yaw):.3f}">'
        f'<geom name="{f.name}" type="box" size="{_f(*f.half)}" pos="0 0 {f.half[2]:.4f}" material="wood" {SURF} friction="0.5 0.02 0.001"/></body>'
        for f in H.FURNITURE)


def bins_xml() -> str:
    out = []
    for b in H.BINS:
        ix, iy = b.inner
        t, h = 0.008, b.height
        s = f'{SURF} friction="0.5 0.02 0.001" rgba="{_f(*b.rgba)}"'
        out.append(f"""
  <body name="{b.name}" pos="{_f(*b.center, 0)}" euler="0 0 {math.degrees(b.yaw):.3f}">
    <geom type="box" size="{_f(ix + t, iy + t, t / 2)}" pos="0 0 {t / 2}" {s}/>
    <geom type="box" size="{_f(t / 2, iy + t, h / 2)}" pos="{_f(ix + t / 2, 0, h / 2)}" {s}/>
    <geom type="box" size="{_f(t / 2, iy + t, h / 2)}" pos="{_f(-ix - t / 2, 0, h / 2)}" {s}/>
    <geom type="box" size="{_f(ix, t / 2, h / 2)}" pos="{_f(0, iy + t / 2, h / 2)}" {s}/>
    <geom type="box" size="{_f(ix, t / 2, h / 2)}" pos="{_f(0, -iy - t / 2, h / 2)}" {s}/>
    <site name="in_{b.name}" type="box" size="{_f(ix, iy, h / 2)}" pos="0 0 {h / 2}" rgba="0 1 0 0"/>
  </body>""")
    return "".join(out)


def machine_xml(mc: H.Machine) -> str:
    """Cabinet with a drum cavity behind a front opening, and a side-hinged door with a handle."""
    hx, hy, hz = mc.half
    o, oz, dep = mc.opening_half, mc.opening_z, mc.drum_depth
    s = f'{SURF} friction="0.4 0.02 0.001" material="appliance"'
    blk = lambda name, x0, x1, y0, y1, z0, z1: (
        f'<geom name="{mc.name}_{name}" type="box" size="{_f((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2)}" '
        f'pos="{_f((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)}" {s}/>')
    geoms = [
        blk("bottom", -hx, hx, -hy, hy, 0, oz - o),
        blk("top", -hx, hx, -hy, hy, oz + o, 2 * hz),
        blk("left", -hx, -o, -hy, hy, oz - o, oz + o),
        blk("right", o, hx, -hy, hy, oz - o, oz + o),
        blk("back", -o, o, -hy, hy - dep, oz - o, oz + o),
    ]
    door_w = o + 0.03
    return f"""
  <body name="{mc.name}" pos="{_f(*mc.center, mc.base_z)}">
    {"".join(geoms)}
    <site name="in_{mc.name}" type="box" size="{_f(o, dep / 2, o)}" pos="{_f(0, hy - dep / 2, oz)}" rgba="0 1 0 0"/>
    <body name="{mc.name}_door" pos="{_f(-door_w, hy + 0.004, oz)}">
      <joint name="{mc.name}_door" axis="0 0 1" range="0 115" limited="true" frictionloss="0.15" damping="0.2"/>
      <geom name="{mc.name}_door" type="box" size="{_f(door_w, 0.02, door_w)}" pos="{_f(door_w, 0.02, 0)}" {SURF} friction="0.4 0.02 0.001" material="glass" mass="2.0"/>
      <geom name="{mc.name}_handle" type="box" size="0.012 0.03 0.06" pos="{_f(2 * door_w - 0.03, 0.05, 0)}" {SURF} friction="0.6 0.02 0.001" material="appliance" mass="0.1"/>
    </body>
  </body>"""


def door_xml(d: H.Door = H.LAUNDRY_DOOR) -> str:
    w, h, t = d.width, d.height, d.thickness
    lo, hi = d.open_range_deg
    lx = w - d.latch_inset
    levers = "".join(f"""
      <body name="lever_{side}" pos="{_f(lx, sy * (t / 2 + 0.025), d.lever_height)}">
        <joint name="lever_{side}" axis="0 {-sy} 0" range="0 50" limited="true" stiffness="{d.lever_spring}" springref="0" damping="0.02"/>
        <geom name="lever_{side}" type="box" size="{_f(d.lever_length / 2, 0.009, 0.009)}" pos="{_f(-d.lever_length / 2, 0, 0)}" {SURF} friction="0.6 0.02 0.001" material="steel" mass="0.15"/>
        <geom type="cylinder" size="0.012 0.012" euler="90 0 0" material="steel" contype="0" conaffinity="0" mass="0.05"/>
      </body>""" for side, sy in (("out", 1), ("in", -1)))
    return f"""
  <body name="{d.name}" pos="{_f(*d.hinge, 0)}">
    <joint name="{d.name}" axis="0 0 1" range="{lo} {hi}" limited="true" frictionloss="{d.hinge_friction}" damping="0.5"/>
    <geom name="{d.name}" type="box" size="{_f(w / 2 - 0.005, t / 2, h / 2)}" pos="{_f(w / 2, 0, h / 2 + 0.01)}" {SURF} friction="0.5 0.02 0.001" material="door" mass="12"/>
    <site name="door_tag" pos="{_f(w / 2, t / 2 + 0.001, 1.3)}" size="0.04" rgba="0 0 0 1"/>
    {levers}
  </body>"""


def dock_xml() -> str:
    d = S.DOCK
    bx, by, bz = d.board_size
    cy, cz = d.contact_size
    target_x = bx + d.spring_travel / 2 + S.GEOM.base_length / 2 + 0.014
    (x, y), yaw = H.DOCK_POSE
    return f"""
  <body name="dock" pos="{_f(x, y, 0)}" euler="0 0 {math.degrees(yaw):.3f}">
    <geom name="dock_board" type="box" size="{_f(bx / 2, by / 2, bz / 2)}" pos="{_f(bx / 2, 0, bz / 2)}" material="dock" {SURF} friction="0.5 0.02 0.001"/>
    <geom name="dock_contact_a" type="box" size="{_f(0.002, cy / 2, cz / 2)}" pos="{_f(bx + 0.002, d.contact_y, S.GEOM.dock_pad_z)}"
          material="copper" {SURF} friction="0.4 0.02 0.001" margin="{d.spring_travel}" gap="{d.spring_travel}"/>
    <geom name="dock_contact_b" type="box" size="{_f(0.002, cy / 2, cz / 2)}" pos="{_f(bx + 0.002, -d.contact_y, S.GEOM.dock_pad_z)}"
          material="copper" {SURF} friction="0.4 0.02 0.001" margin="{d.spring_travel}" gap="{d.spring_travel}"/>
    <geom name="dock_led" type="sphere" size="0.012" pos="{_f(bx + 0.004, 0, d.beacon_z)}" material="ir" contype="0" conaffinity="0" group="3"/>
    <site name="beacon" pos="{_f(bx + 0.012, 0, d.beacon_z)}" size="0.005"/>
    <site name="dock_target" pos="{_f(target_x, 0, 0)}" size="0.01" rgba="0 1 0 0.3"/>
  </body>"""


def chair_xml() -> str:
    """Rolling desk chair: 12 kg, five casters that roll (very low friction)."""
    x, y = H.CHAIR_HOME
    legs, casters = [], []
    for k in range(5):
        a = 2 * math.pi * k / 5
        cx, cy = 0.3 * math.cos(a), 0.3 * math.sin(a)
        legs.append(f'<geom type="capsule" fromto="0 0 0.07 {cx:.3f} {cy:.3f} 0.06" size="0.015" mass="0.5" material="servo" {SURF} friction="0.5 0.02 0.001"/>')
        casters.append(f'<geom name="chair_caster_{k}" type="sphere" size="0.025" pos="{cx:.3f} {cy:.3f} 0.025" mass="0.1" material="servo" priority="3" condim="3" friction="0.03 0.001 0.001"/>')
    return f"""
  <body name="chair" pos="{_f(x, y, 0.002)}">
    <freejoint name="chair"/>
    <geom name="chair_seat" type="box" size="0.24 0.24 0.04" pos="0 0 0.48" mass="4" material="fabric" {SURF} friction="0.6 0.02 0.001"/>
    <geom name="chair_back" type="box" size="0.03 0.22 0.22" pos="-0.21 0 0.76" mass="2" material="fabric" {SURF} friction="0.6 0.02 0.001"/>
    <geom type="cylinder" size="0.03 0.17" pos="0 0 0.27" mass="1.5" material="steel" {SURF} friction="0.5 0.02 0.001"/>
    {"".join(legs)}
    {"".join(casters)}
  </body>"""


def room_floors_xml() -> str:
    """Fixed room floors laid over the heightfield (carpet in the bedroom, tile in the laundry room).
    The env levels the heightfield under them and picks their friction each episode."""
    out = []
    for room, kind in H.ROOM_FLOOR.items():
        if kind == "hardwood":            # the heightfield itself is the living-room floor
            continue
        x0, y0, x1, y1 = H.ROOMS[room]
        thick = 0.008 if kind == "carpet" else 0.002
        out.append(f'<body name="floor_{room}" pos="{_f((x0 + x1) / 2, (y0 + y1) / 2, 0)}">'
                   f'<geom name="floor_{room}" type="box" size="{_f((x1 - x0) / 2, (y1 - y0) / 2, thick / 2)}" '
                   f'pos="0 0 {thick / 2}" priority="2" condim="4" friction="0.6 0.02 0.001" rgba="0.5 0.5 0.5 1"/></body>')
    return "\n    ".join(out)


def scene_xml(servo_mode: str = S.SERVO_MODE) -> str:
    return f"""<mujoco model="homebody">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{1.0 / S.PHYSICS_HZ}" integrator="implicitfast" cone="elliptic"/>
  <size memory="80M"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55"/>
    <quality shadowsize="4096"/>
    <map znear="0.001"/>
  </visual>
  {R.defaults_xml()}
  <asset>
    {R.materials_xml()}
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.96 0.95 0.92" rgb2="0.80 0.80 0.78" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.80 0.76" rgb2="0.76 0.74 0.70" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="60 40" reflectance="0.05"/>
    {T.assets_xml(H.EXTENT)}
    <material name="wall" rgba="0.93 0.91 0.86 1"/>
    <material name="wood" rgba="0.55 0.38 0.24 1"/>
    <material name="door" rgba="0.85 0.80 0.70 1"/>
    <material name="fabric" rgba="0.20 0.22 0.28 1"/>
    <material name="appliance" rgba="0.96 0.96 0.96 1"/>
    <material name="glass" rgba="0.45 0.55 0.60 0.6"/>
    <material name="dock" rgba="0.25 0.27 0.30 1"/>
    <material name="copper" rgba="0.85 0.50 0.25 1" reflectance="0.3"/>
    <material name="ir" rgba="0.6 0.0 0.1 1" emission="0.5"/>
  </asset>
  <worldbody>
    <light pos="-2.7 1.2 2.3" dir="0 0 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <light pos="1.0 0 2.3" dir="0 0 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <light pos="3.7 -2.1 2.3" dir="0 0 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <camera name="overview" pos="0 -5.2 10.5" xyaxes="1 0 0 0 0.9 0.44"/>
    {look_at("laundry", (3.15, -1.35, 2.1), (4.1, -2.5, 0.6), 80)}
    {look_at("bedroom", (-1.3, -0.3, 2.2), (-3.0, 1.6, 0.3), 75)}
    {look_at("living", (-0.8, -2.8, 2.2), (1.8, 0.8, 0.3), 75)}
    {look_at("door", (2.3, 0.3, 1.9), (3.6, -1.05, 0.8), 70)}
    {look_at("demo_pick", (0.75, -1.55, 1.35), (-0.55, -0.55, 0.25), 60)}
    {T.world_xml()}
    {room_floors_xml()}
    {walls_xml()}
    {furniture_xml()}
    {bins_xml()}
    {machine_xml(H.WASHER)}
    {machine_xml(H.DRYER)}
    {door_xml()}
    {dock_xml()}
    {R.body_xml()}
    {chair_xml()}
    {I.items_xml()}
  </worldbody>
  <equality>
    {R.equality_xml()}
    <joint joint1="lever_in" joint2="lever_out" polycoef="0 1 0 0 0" solref="0.005 1"/>
  </equality>
  <actuator>
    {R.actuators_xml(servo_mode)}
  </actuator>
  <sensor>
    {R.sensors_xml()}
  </sensor>
</mujoco>"""


def build_model(servo_mode: str = S.SERVO_MODE):
    import mujoco
    return mujoco.MjModel.from_xml_string(scene_xml(servo_mode))
