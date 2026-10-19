"""The world: a room with furniture, the charging dock and a real floor.

Static bodies (furniture, dock, floor patches) are moved per episode by ``env``
through ``model.body_pos`` and the floor heightfield is regenerated
(``terrain``), so one compiled model covers every layout.

Every world geom has contact priority 2 and the robot's have 0, so friction
and softness of a contact are the surface's (see ``terrain``).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import robot_xml as R
from . import terrain as T
from . import spec as S


@dataclass(frozen=True)
class RoomSpec:
    size: tuple = (6.0, 5.0)          # interior x, y (m)
    wall_height: float = 0.8
    wall_thickness: float = 0.05
    n_furniture: int = 6               # boxes: sofa, table, shelves, boxes


ROOM = RoomSpec()

# half-sizes of the furniture boxes: sofa, coffee table, shelf, storage box, armchair, crate
FURNITURE_SIZES = [
    (0.45, 0.9, 0.22), (0.35, 0.6, 0.2), (0.2, 0.5, 0.45), (0.2, 0.2, 0.15), (0.4, 0.4, 0.22), (0.15, 0.15, 0.1),
]


SURF = 'priority="2" condim="3"'


def _f(*v):
    return " ".join(f"{x:.5g}" for x in v)


def dock_xml() -> str:
    d = S.DOCK
    bx, by, bz = d.base_size
    px, py = d.plate_size
    off = (d.plate_gap + py) / 2
    return f"""
  <body name="dock" pos="1 0 0">
    <geom name="dock_base" type="box" size="{_f(bx / 2, by / 2, bz / 2)}" pos="0 0 {bz / 2:.4f}" material="dock" {SURF} friction="0.6 0.02 0.001"/>
    <geom name="dock_plate_a" type="box" size="{_f(px / 2, py / 2, 0.001)}" pos="0.02 {off:.4f} {bz + 0.001:.4f}" material="copper" {SURF} friction="0.6 0.02 0.001"/>
    <geom name="dock_plate_b" type="box" size="{_f(px / 2, py / 2, 0.001)}" pos="0.02 {-off:.4f} {bz + 0.001:.4f}" material="copper" {SURF} friction="0.6 0.02 0.001"/>
    <geom name="dock_post" type="box" size="0.02 0.04 {d.beacon_height / 2:.4f}" pos="{-bx / 2 + 0.02:.4f} 0 {d.beacon_height / 2:.4f}" material="dock" {SURF} friction="0.5 0.02 0.001"/>
    <geom name="dock_led" type="sphere" size="0.012" pos="{-bx / 2 + 0.045:.4f} 0 {d.beacon_height - 0.03:.4f}" material="ir" contype="0" conaffinity="0" group="3"/>
    <site name="beacon" pos="{-bx / 2 + 0.045:.4f} 0 {d.beacon_height - 0.03:.4f}" size="0.005"/>
    <site name="dock_target" pos="0.02 0 {bz:.4f}" size="0.01" rgba="0 1 0 0.3"/>
  </body>"""


def room_xml(room: RoomSpec = ROOM) -> str:
    X, Y = room.size[0] / 2, room.size[1] / 2
    t, h = room.wall_thickness, room.wall_height
    walls = "".join([
        f'<geom name="wall_n" type="box" size="{_f(X + t, t / 2, h / 2)}" pos="{_f(0, Y + t / 2, h / 2)}" material="wall" {SURF} friction="0.5 0.02 0.001"/>',
        f'<geom name="wall_s" type="box" size="{_f(X + t, t / 2, h / 2)}" pos="{_f(0, -Y - t / 2, h / 2)}" material="wall" {SURF} friction="0.5 0.02 0.001"/>',
        f'<geom name="wall_e" type="box" size="{_f(t / 2, Y, h / 2)}" pos="{_f(X + t / 2, 0, h / 2)}" material="wall" {SURF} friction="0.5 0.02 0.001"/>',
        f'<geom name="wall_w" type="box" size="{_f(t / 2, Y, h / 2)}" pos="{_f(-X - t / 2, 0, h / 2)}" material="wall" {SURF} friction="0.5 0.02 0.001"/>',
    ])
    furniture = "".join(
        f'<body name="furniture_{k}" pos="{2 + k} 2 0"><geom name="furniture_{k}" type="box" '
        f'size="{_f(*FURNITURE_SIZES[k % len(FURNITURE_SIZES)])}" '
        f'pos="0 0 {FURNITURE_SIZES[k % len(FURNITURE_SIZES)][2]:.3f}" material="wood" {SURF} friction="0.5 0.02 0.001"/></body>'
        for k in range(room.n_furniture)
    )
    return f"""
  {T.world_xml()}
  {walls}
  {furniture}"""


def scene_xml(room: RoomSpec = ROOM) -> str:
    return f"""<mujoco model="homebody">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{1.0 / S.PHYSICS_HZ}" integrator="implicitfast" cone="elliptic"/>
  <size memory="20M"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/>
    <quality shadowsize="4096"/>
  </visual>
  {R.defaults_xml()}
  <asset>
    {R.materials_xml()}
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.96 0.95 0.92" rgb2="0.80 0.80 0.78" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.80 0.76" rgb2="0.76 0.74 0.70" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="40 32" reflectance="0.05"/>
    {T.assets_xml(room.size)}
    <material name="wall" rgba="0.93 0.91 0.86 1"/>
    <material name="wood" rgba="0.55 0.38 0.24 1"/>
    <material name="rug" rgba="0.55 0.22 0.22 1"/>
    <material name="dock" rgba="0.25 0.27 0.30 1"/>
    <material name="copper" rgba="0.85 0.50 0.25 1" reflectance="0.3"/>
    <material name="ir" rgba="0.6 0.0 0.1 1" emission="0.5"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.5 0.5 0.5" castshadow="true"/>
    <camera name="overview" pos="0 -3.6 6.0" xyaxes="1 0 0 0 0.86 0.51"/>
    {room_xml(room)}
    {dock_xml()}
    {R.body_xml()}
    {T.clutter_xml()}
  </worldbody>
  <actuator>
    {R.actuators_xml()}
  </actuator>
  <sensor>
    {R.sensors_xml()}
  </sensor>
</mujoco>"""


def build_model(room: RoomSpec = ROOM):
    import mujoco
    model = mujoco.MjModel.from_xml_string(scene_xml(room))
    R.joint_param_overrides(model)
    return model
