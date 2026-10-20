"""MJCF for the owner's recreated home (``home``), the robot (``robot_xml``), the floor (``terrain``)
and items.

Walls are boxes generated around the rooms; furniture is one body per piece
(so movable pieces can shift each episode); the floor and the living-room rug
use textures cut from the scan. ``Look`` varies colours and lighting per
episode, so camera-based skills learn the room rather than one exact render.

Every world geom has contact priority 2 and the robot's wheels and body 0, so
friction and softness of a contact are the surface's (see ``terrain``).
"""
from __future__ import annotations

import math
import os

import numpy as np

from . import home as H
from . import items as I
from . import robot_xml as R
from . import spec as S
from . import terrain as T

SURF = 'priority="2" condim="3"'
ROOT = os.path.join(os.path.dirname(__file__), "..")


def _f(*v):
    return " ".join(f"{x:.5g}" for x in v)


def look_at(name, pos, target, fovy=60):
    """A fixed camera at ``pos`` aimed at ``target``."""
    p, t = np.array(pos, float), np.array(target, float)
    fwd = (t - p) / np.linalg.norm(t - p)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return f'<camera name="{name}" pos="{_f(*p)}" xyaxes="{_f(*right, *up)}" fovy="{fovy}"/>'


def bins_xml() -> str:
    out = []
    for b in H.BINS:
        ix, iy = b.inner
        t, h = 0.008, b.height
        s = f'{SURF} friction="0.5 0.02 0.001" rgba="{_f(*b.rgba)}"'
        walls = "" if h < 0.03 else f"""
    <geom type="box" size="{_f(t / 2, iy + t, h / 2)}" pos="{_f(ix + t / 2, 0, h / 2)}" {s}/>
    <geom type="box" size="{_f(t / 2, iy + t, h / 2)}" pos="{_f(-ix - t / 2, 0, h / 2)}" {s}/>
    <geom type="box" size="{_f(ix, t / 2, h / 2)}" pos="{_f(0, iy + t / 2, h / 2)}" {s}/>
    <geom type="box" size="{_f(ix, t / 2, h / 2)}" pos="{_f(0, -iy - t / 2, h / 2)}" {s}/>"""
        zone_h = max(h, 0.25)         # flat zones (shoe rack, recycling spot) count what rests on them
        out.append(f"""
  <body name="{b.name}" pos="{_f(*b.center, b.z)}" euler="0 0 {math.degrees(b.yaw):.3f}">
    <geom type="box" size="{_f(ix + t, iy + t, t / 2)}" pos="0 0 {t / 2}" {s}/>{walls}
    <site name="in_{b.name}" type="box" size="{_f(ix, iy, zone_h / 2)}" pos="0 0 {zone_h / 2}" rgba="0 1 0 0"/>
  </body>""")
    return "".join(out)


def zones_xml() -> str:
    out = []
    for name, (x0, y0, x1, y1), (z0, z1) in H.ZONES:
        (a, b), (c, d) = H.A(x0, y0), H.A(x1, y1)
        out.append(f'<site name="in_{name}" type="box" size="{_f((c - a) / 2, (d - b) / 2, (z1 - z0) / 2)}" '
                   f'pos="{_f((a + c) / 2, (b + d) / 2, (z0 + z1) / 2)}" rgba="0 1 0 0"/>')
    return "\n    ".join(out)


def machine_xml(mc: H.Machine) -> str:
    """Cabinet with a drum cavity behind a front opening (local +y), and a side-hinged door with a handle."""
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
  <body name="{mc.name}" pos="{_f(*mc.center, mc.base_z)}" euler="0 0 {math.degrees(mc.yaw):.3f}">
    {"".join(geoms)}
    <site name="in_{mc.name}" type="box" size="{_f(o, dep / 2, o)}" pos="{_f(0, hy - dep / 2, oz)}" rgba="0 1 0 0"/>
    <body name="{mc.name}_door" pos="{_f(-door_w, hy + 0.004, oz)}">
      <joint name="{mc.name}_door" axis="0 0 1" range="0 115" limited="true" frictionloss="0.15" damping="0.2"/>
      <geom name="{mc.name}_door" type="box" size="{_f(door_w, 0.02, door_w)}" pos="{_f(door_w, 0.02, 0)}" {SURF} friction="0.4 0.02 0.001" material="glass" mass="2.0"/>
      <geom name="{mc.name}_handle" type="box" size="0.012 0.03 0.06" pos="{_f(2 * door_w - 0.03, 0.05, 0)}" {SURF} friction="0.6 0.02 0.001" material="appliance" mass="0.1"/>
    </body>
  </body>"""


def door_xml(d: H.Door) -> str:
    """A door panel along local +x from the hinge. Lever doors get a lever on each face (coupled by an
    equality constraint); catch doors get a knob on each face."""
    w, h, t = d.width, d.height, d.thickness
    lx = w - d.latch_inset
    if d.latch == "lever":
        handles = "".join(f"""
      <body name="{d.name}_lever_{side}" pos="{_f(lx, sy * (t / 2 + 0.025), d.handle_height)}">
        <joint name="{d.name}_lever_{side}" axis="0 {-sy} 0" range="0 50" limited="true" stiffness="{d.lever_spring}" springref="0" damping="0.02"/>
        <geom name="{d.name}_lever_{side}" type="box" size="{_f(d.lever_length / 2, 0.009, 0.009)}" pos="{_f(-d.lever_length / 2, 0, 0)}" {SURF} friction="0.6 0.02 0.001" material="steel" mass="0.15"/>
        <geom type="cylinder" size="0.012 0.012" euler="90 0 0" material="steel" contype="0" conaffinity="0" mass="0.05"/>
      </body>""" for side, sy in (("a", 1), ("b", -1)))
    else:
        handles = "".join(
            f'<geom name="{d.name}_knob_{side}" type="capsule" fromto="{_f(lx, sy * t / 2, d.handle_height, lx, sy * (t / 2 + 0.05), d.handle_height)}" '
            f'size="0.012" {SURF} friction="0.6 0.02 0.001" material="steel" mass="0.05"/>'
            for side, sy in (("a", 1), ("b", -1)))
    return f"""
  <body name="{d.name}" pos="{_f(*d.hinge, 0)}" euler="0 0 {math.degrees(d.yaw):.3f}">
    <joint name="{d.name}" axis="0 0 1" range="0 {d.open_deg}" limited="true" frictionloss="{d.hinge_friction}" damping="0.5"/>
    <geom name="{d.name}" type="box" size="{_f(w / 2 - 0.005, t / 2, h / 2)}" pos="{_f(w / 2, 0, h / 2 + 0.01)}" {SURF} friction="0.5 0.02 0.001" material="door" mass="10"/>
    {handles}
  </body>"""


def doors_equality_xml() -> str:
    return "\n    ".join(
        f'<joint joint1="{d.name}_lever_b" joint2="{d.name}_lever_a" polycoef="0 1 0 0 0" solref="0.005 1"/>'
        for d in H.DOORS if d.latch == "lever")


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
        cx, cy = 0.28 * math.cos(a), 0.28 * math.sin(a)
        legs.append(f'<geom type="capsule" fromto="0 0 0.07 {cx:.3f} {cy:.3f} 0.06" size="0.015" mass="0.5" material="servo" {SURF} friction="0.5 0.02 0.001"/>')
        casters.append(f'<geom name="chair_caster_{k}" type="sphere" size="0.025" pos="{cx:.3f} {cy:.3f} 0.025" mass="0.1" material="servo" priority="3" condim="3" friction="0.03 0.001 0.001"/>')
    return f"""
  <body name="chair" pos="{_f(x, y, 0.002)}">
    <freejoint name="chair"/>
    <geom name="chair_seat" type="box" size="0.22 0.22 0.04" pos="0 0 0.48" mass="4" material="fabric" {SURF} friction="0.6 0.02 0.001"/>
    <geom name="chair_back" type="box" size="0.03 0.2 0.2" pos="-0.19 0 0.74" mass="2" material="fabric" {SURF} friction="0.6 0.02 0.001"/>
    <geom type="cylinder" size="0.03 0.17" pos="0 0 0.27" mass="1.5" material="steel" {SURF} friction="0.5 0.02 0.001"/>
    {"".join(legs)}
    {"".join(casters)}
  </body>"""


def walls_xml() -> str:
    return "\n    ".join(
        f'<geom name="wall_{k}" type="box" size="{_f(hx, hy, H.WALL_H / 2)}" pos="{_f(cx, cy, H.WALL_H / 2)}" '
        f'material="m_wall" {SURF} friction="0.5 0.02 0.001"/>'
        for k, (cx, cy, hx, hy) in enumerate(H.WALLS))


def furniture_xml() -> str:
    """One body per furniture group, so a group can be shifted as a whole."""
    groups = {}
    for p in H.FURNITURE:
        groups.setdefault(p.group or p.name, []).append(p)
    out = []
    for g, parts in groups.items():
        geoms = []
        for p in parts:
            x0, y0, x1, y1 = H.AR(*p.rect)
            pos = _f((x0 + x1) / 2, (y0 + y1) / 2, (p.z0 + p.z1) / 2)
            size = (_f(min(x1 - x0, y1 - y0) / 2, (p.z1 - p.z0) / 2) if p.shape == "cylinder"
                    else _f((x1 - x0) / 2, (y1 - y0) / 2, (p.z1 - p.z0) / 2))
            geoms.append(f'<geom name="{p.name}" type="{p.shape}" size="{size}" pos="{pos}" material="m_{p.material}" '
                         f'{SURF} friction="0.6 0.02 0.001"/>')
        out.append(f'<body name="furn_{g}">{"".join(geoms)}</body>')
    return "\n    ".join(out)


def rug_xml() -> str:
    """The living-room rug ("KEEP OFF"), fixed in place, with its texture from the scan."""
    (a, b), (c, d) = H.A(1.40, 1.90), H.A(2.70, 3.50)
    return (f'<body name="rug_living" pos="{_f((a + c) / 2, (b + d) / 2, 0)}"><geom name="rug_living" type="box" '
            f'size="{_f((c - a) / 2, (d - b) / 2, 0.003)}" pos="0 0 0.003" priority="2" condim="4" friction="0.9 0.02 0.001" '
            f'solref="0.035 1" material="m_rug"/></body>')


def materials_xml() -> str:
    tex = H.TEXTURES.replace("\\", "/")
    out = [f'<texture name="t_wood" type="2d" file="{tex}/floor_wood.png"/>',
           f'<texture name="t_rug" type="2d" file="{tex}/rug.png"/>',
           '<material name="floor" texture="t_wood" texuniform="true" texrepeat="1 1.43" reflectance="0.04"/>',
           '<material name="m_rug" texture="t_rug" texrepeat="1 1"/>']
    out += [f'<material name="m_{n}" rgba="{_f(*c)}"/>' for n, c in H.MATERIALS.items()]
    return "\n    ".join(out)


class Look:
    """Per-episode appearance randomisation: material colours, light levels, movable furniture positions."""

    def __init__(self, model, home):
        import mujoco
        self.m = model
        self.mats = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, f"m_{n}") for n in home.MATERIALS}
        self.base = {n: np.array(c, float) for n, c in home.MATERIALS.items()}
        self.lights = list(range(model.nlight))
        self.light_base = model.light_diffuse.copy()
        self.movable = {}
        for g, jit in home.MOVABLE.items():
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"furn_{g}")
            if b >= 0:
                self.movable[b] = (model.body_pos[b].copy(), jit)

    def randomize(self, rng):
        m = self.m
        if rng is None:
            for n, mid in self.mats.items():
                m.mat_rgba[mid] = self.base[n]
            m.light_diffuse[:] = self.light_base
            for b, (p0, _) in self.movable.items():
                m.body_pos[b] = p0
            return
        for n, mid in self.mats.items():
            c = self.base[n].copy()
            c[:3] = np.clip(c[:3] * rng.uniform(0.8, 1.15) + rng.normal(0, 0.03, 3), 0, 1)
            m.mat_rgba[mid] = c
        m.light_diffuse[:] = self.light_base * rng.uniform(0.6, 1.4)
        for b, (p0, jit) in self.movable.items():
            m.body_pos[b] = p0 + np.array([*rng.uniform(-jit, jit, 2), 0])


def scene_xml(servo_mode: str = S.SERVO_MODE) -> str:
    A = H.A
    return f"""<mujoco model="homebody">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{1.0 / S.PHYSICS_HZ}" integrator="implicitfast" cone="elliptic"/>
  <size memory="120M"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.35 0.35 0.35"/>
    <quality shadowsize="4096"/>
    <map znear="0.001" zfar="80"/>
  </visual>
  {R.defaults_xml()}
  <asset>
    {R.materials_xml()}
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.96 0.95 0.92" rgb2="0.80 0.80 0.78" width="256" height="256"/>
    {materials_xml()}
    {T.assets_xml(H.EXTENT)}
    <material name="door" rgba="0.93 0.92 0.88 1"/>
    <material name="fabric" rgba="0.70 0.60 0.45 1"/>
    <material name="appliance" rgba="0.96 0.96 0.96 1"/>
    <material name="glass" rgba="0.45 0.55 0.60 0.6"/>
    <material name="dock" rgba="0.25 0.27 0.30 1"/>
    <material name="copper" rgba="0.85 0.50 0.25 1" reflectance="0.3"/>
    <material name="ir" rgba="0.6 0.0 0.1 1" emission="0.5"/>
  </asset>
  <worldbody>
    <light pos="{_f(*A(-3.9, 2.5), 2.35)}" dir="0 0 -1" diffuse="0.45 0.43 0.40" castshadow="true" attenuation="0.5 0.2 0"/>
    <light pos="{_f(*A(2.2, 2.6), 2.35)}" dir="0 0 -1" diffuse="0.50 0.50 0.48" castshadow="true" attenuation="0.5 0.2 0"/>
    <light pos="{_f(*A(2.2, -0.9), 2.35)}" dir="0 0 -1" diffuse="0.45 0.45 0.45" castshadow="false" attenuation="0.5 0.2 0"/>
    <light pos="{_f(*A(1.9, -3.6), 2.35)}" dir="0 0 -1" diffuse="0.40 0.40 0.40" castshadow="false" attenuation="0.5 0.2 0"/>
    <light pos="{_f(*A(-1.0, 0.5), 2.35)}" dir="0 0 -1" diffuse="0.30 0.30 0.30" castshadow="false" attenuation="0.5 0.2 0"/>
    <camera name="overview" pos="0 -7.5 14" xyaxes="1 0 0 0 0.88 0.47"/>
    {look_at("bedroom", (*A(-2.75, 0.25), 2.0), (*A(-4.2, 2.6), 0.3), 80)}
    {look_at("closet", (*A(-3.0, 1.6), 1.9), (*A(-4.6, 0.5), 0.3), 75)}
    {look_at("laundry", (*A(1.4, -2.6), 1.9), (*A(2.9, -3.9), 0.8), 75)}
    {look_at("kitchen", (*A(0.9, 0.9), 1.9), (*A(1.6, -0.6), 0.4), 75)}
    {look_at("living", (*A(0.9, 1.3), 1.9), (*A(3.2, 3.6), 0.3), 80)}
    {look_at("hallway", (*A(0.4, 0.5), 1.7), (*A(-3.0, 0.5), 0.5), 70)}
    {T.world_xml()}
    {rug_xml()}
    {walls_xml()}
    {furniture_xml()}
    {zones_xml()}
    {bins_xml()}
    {machine_xml(H.WASHER)}
    {machine_xml(H.DRYER)}
    {"".join(door_xml(d) for d in H.DOORS)}
    {dock_xml()}
    {R.body_xml()}
    {chair_xml()}
    {I.items_xml()}
  </worldbody>
  <equality>
    {R.equality_xml()}
    {doors_equality_xml()}
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
