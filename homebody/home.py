"""The owner's home: floor plan, furniture, fixtures and containers.

A plausible stand-in until the real rooms are scanned (phone lidar -> mesh):
a bedroom (bed, desk, rolling desk chair, wardrobe, trash can, laundry basket,
the charging dock), an open living room and kitchen, and a laundry room
behind a lever-handle door with a stacked washer and dryer and a bucket of
laundry pods.

Coordinates: metres, x east, y north, origin at the centre of the apartment.
Everything that has to be measured in the real home is in this file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EXTENT = (9.0, 6.0)                       # apartment bounding box, x by y
WALL_T = 0.10
WALL_H = 2.40

# accessible floor, as rectangles (x0, y0, x1, y1); the bathroom (south-west) is shut
ROOMS = {
    "bedroom": (-4.5, -0.5, -1.0, 3.0),
    "living": (-1.0, -3.0, 3.0, 3.0),
    "kitchen": (3.0, -1.2, 4.5, 3.0),
    "laundry": (3.0, -3.0, 4.5, -1.2),
}
ROOM_FLOOR = {"bedroom": "carpet", "living": "hardwood", "kitchen": "tile", "laundry": "tile"}

# interior wall segments (x0, y0, x1, y1), doorways left as gaps
WALLS = [
    (-1.0, -0.5, -1.0, 1.0), (-1.0, 1.85, -1.0, 3.0),     # bedroom east wall, doorway y 1.0..1.85
    (-4.5, -0.5, -1.0, -0.5),                             # bedroom south wall (bathroom behind)
    (-1.0, -3.0, -1.0, -0.5),                             # bathroom east wall
    (3.0, -1.2, 3.1, -1.2), (3.92, -1.2, 4.5, -1.2),      # laundry north wall, doorway x 3.1..3.92
    (3.0, -3.0, 3.0, -1.2),                               # laundry west wall
]
DOORWAYS = {"bedroom": ((-1.0, 1.425), math.pi / 2), "laundry": ((3.51, -1.2), 0.0)}   # centre, sill yaw


@dataclass(frozen=True)
class Box:
    """Solid furniture: centre (x, y), half-sizes (x, y, z), yaw. Sits on the floor."""
    name: str
    center: tuple
    half: tuple
    yaw: float = 0.0
    room: str = ""
    jitter: float = 0.0          # m of random placement per episode


FURNITURE = [
    Box("bed", (-3.545, 1.69), (0.955, 0.685, 0.275), room="bedroom"),
    Box("desk", (-1.75, 2.69), (0.60, 0.30, 0.375), room="bedroom"),
    Box("wardrobe", (-3.0, -0.19), (0.60, 0.30, 1.00), room="bedroom"),
    Box("kitchen_counter", (2.5, 2.69), (1.90, 0.30, 0.45), room="kitchen"),
    Box("fridge", (0.15, 2.64), (0.40, 0.35, 0.90), room="living"),
    Box("sofa", (0.55, -2.54), (1.05, 0.45, 0.22), room="living", jitter=0.05),
    Box("coffee_table", (0.55, -1.30), (0.50, 0.30, 0.22), room="living", jitter=0.15),
    Box("dining_table", (2.0, 0.6), (0.60, 0.45, 0.375), room="living", jitter=0.15),
]


@dataclass(frozen=True)
class Bin:
    """An open container: centre (x, y), inner half-sizes (x, y), wall height, yaw."""
    name: str
    center: tuple
    inner: tuple
    height: float
    yaw: float = 0.0
    jitter: float = 0.0
    rgba: tuple = (0.3, 0.6, 0.65, 1)


BINS = [
    Bin("trash_can", (-1.30, 2.15), (0.13, 0.13), 0.36, jitter=0.1, rgba=(0.25, 0.25, 0.28, 1)),
    Bin("laundry_basket", (-2.15, 0.05), (0.24, 0.17), 0.30, jitter=0.15, rgba=(0.85, 0.85, 0.80, 1)),
    Bin("pod_bucket", (3.30, -2.78), (0.09, 0.09), 0.18, rgba=(0.95, 0.50, 0.15, 1)),
]


@dataclass(frozen=True)
class Machine:
    """A front-loading machine: cabinet footprint centre, half-sizes, base height, door on the front (+y side
    when yaw=0), drum opening (half-size) at ``opening_z`` above the cabinet base."""
    name: str
    center: tuple
    half: tuple = (0.345, 0.375, 0.425)
    base_z: float = 0.0
    opening_half: float = 0.18
    opening_z: float = 0.45
    drum_depth: float = 0.50


# stacked washer (bottom) and dryer (top) against the laundry room's south wall, doors facing north
WASHER = Machine("washer", (4.045, -2.625), base_z=0.0)
DRYER = Machine("dryer", (4.045, -2.625), base_z=0.85)


@dataclass(frozen=True)
class Door:
    """Hinged interior door with lever handles on both sides. Hinge at ``hinge``; closed, the panel runs
    along +x for ``width``; it swings toward -y (into the laundry room)."""
    name: str = "laundry_door"
    hinge: tuple = (3.10, -1.20)
    width: float = 0.80
    height: float = 2.0
    thickness: float = 0.035
    open_range_deg: tuple = (-100.0, 0.0)
    hinge_friction: float = 0.6       # N m
    lever_height: float = 0.95
    lever_length: float = 0.12
    lever_unlatch_deg: float = 30.0   # press the lever this far down to free the latch
    lever_spring: float = 1.5         # N m/rad return spring (about 1 N m to fully press: lever torque ~ ADA limit)
    latch_inset: float = 0.07         # lever pivot distance from the free edge


LAUNDRY_DOOR = Door()

DOCK_POSE = ((-1.05, 0.30), math.pi)      # on the bedroom's east wall, facing west into the bedroom

CHAIR_HOME = (-1.75, 2.05)


def obstacle_rects(extra=()):
    """2D rectangles (cx, cy, hx, hy, yaw) that block floor space: walls, furniture, machines, bins."""
    rects = []
    for x0, y0, x1, y1 in WALLS:
        horiz = abs(y1 - y0) < 1e-9
        rects.append(((x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2 if horiz else WALL_T / 2,
                      WALL_T / 2 if horiz else abs(y1 - y0) / 2, 0.0))
    for f in FURNITURE:
        rects.append((*f.center, f.half[0], f.half[1], f.yaw))
    rects.append((*WASHER.center, WASHER.half[0], WASHER.half[1], 0.0))
    rects.extend(extra)
    return rects


def _dist_to_rect(p, r):
    cx, cy, hx, hy, yaw = r
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = p[0] - cx, p[1] - cy
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    ex, ey = max(abs(lx) - hx, 0.0), max(abs(ly) - hy, 0.0)
    return math.hypot(ex, ey)


def is_free(p, radius, rects, rooms=None) -> bool:
    """A circle of ``radius`` at ``p`` lies inside one accessible room and clear of every rectangle."""
    rooms = rooms or ROOMS
    inside = any(x0 + radius <= p[0] <= x1 - radius and y0 + radius <= p[1] <= y1 - radius
                 for x0, y0, x1, y1 in rooms.values())
    return inside and all(_dist_to_rect(p, r) > radius for r in rects)


def room_of(p):
    for name, (x0, y0, x1, y1) in ROOMS.items():
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
            return name
    return None
