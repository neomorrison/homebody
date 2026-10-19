"""The owner's home, from their phone LiDAR scan (2026-10-19).

The scan (scripts/import_scan.py -> assets/home_scan.npz) gives the static
shape of the apartment as a height map: walls, bed, desk, sofa, counters. The
owner's loose clutter is stripped out of that map here (STRIP and the edit
lists) and comes back as movable items in ``items``, so the robot can pick it
up. Machines, doors and containers that the robot must operate are replaced
by simulated versions at their measured positions.

Coordinates: metres, sim frame (x east, y north, origin at the centre of the
scan). Positions below are written in the scan's aligned frame with ``A()``,
so they can be checked against the scan's top-down render.

Rooms (aligned frame): bedroom with a walk-in closet (west), a 1 m hallway,
living room (north-east), kitchen and foyer (east), the laundry closet with a
stacked 24 in washer and dryer (south-east), a back room (south).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

_SCAN = os.path.join(os.path.dirname(__file__), "assets", "home_scan.npz")
_raw = np.load(_SCAN)
RES = float(_raw["res"])
_X0, _Y0 = float(_raw["x0"]), float(_raw["y0"])
_NY, _NX = _raw["H"].shape
CX, CY = _X0 + _NX * RES / 2, _Y0 + _NY * RES / 2      # aligned-frame centre = sim origin
EXTENT = (_NX * RES, _NY * RES)
SCAN_ROT = float(_raw["rot"])
SCAN_FLOOR = float(_raw["floor"])
SCAN_SOURCE = str(_raw["source"])
WALL_H = 2.10                                           # map height for walls and unseen space


def A(x, y):
    """Aligned scan frame -> sim frame."""
    return (x - CX, y - CY)


def AR(x0, y0, x1, y1):
    """Aligned rectangle -> sim rectangle (x0, y0, x1, y1)."""
    a, b = A(x0, y0), A(x1, y1)
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


# rooms, for labels and spawning (the map itself decides what is free floor)
ROOMS = {
    "closet": AR(-5.28, 0.02, -4.15, 1.55),
    "bedroom": AR(-4.15, 0.02, -2.62, 4.25),
    "hallway": AR(-2.62, 0.02, 0.62, 1.0),
    "living": AR(0.5, 1.0, 4.05, 4.65),
    "kitchen": AR(0.62, -2.0, 4.05, 1.0),
    "foyer": AR(0.62, -3.45, 3.45, -2.0),
    "laundry": AR(2.42, -4.45, 3.35, -3.45),
    "back_room": AR(1.3, -7.0, 3.4, -4.45),
    "west_hall": AR(-2.7, -4.25, 0.62, -3.25),
}
ROOM_FLOOR: dict = {}      # every room is the same wood-look floor; the heightfield's surface covers it


# ----------------------------------------------------------------------------
# turning the scan into a clean static map
# ----------------------------------------------------------------------------

STRIP_BELOW = 0.32         # anything lower than this on the floor is clutter, not structure
STRIP_BLOB = 0.12          # m^2: smaller free-standing things are clutter too (bags, backpacks)
CLEAR = [                  # aligned rects set to floor: the real clutter, and what the sim replaces
    (-5.26, 0.02, -4.16, 1.0),      # closet floor (cardboard box, bags); basket and shoe rack go here
    (-5.26, 1.6, -3.95, 2.16),      # pile at the foot of the bed (the pillows start here)
    (-3.35, 3.1, -2.85, 3.8),       # desk chair (simulated, it rolls)
    (2.40, -4.43, 3.33, -3.47),     # laundry closet: machines and folding door (simulated)
    (1.2, -0.25, 1.56, 0.12),       # kitchen trash can (simulated)
    (-2.70, 0.06, -2.52, 0.97),     # bedroom door leaf, scanned shut across the doorway (simulated)
    (1.45, -4.44, 2.39, -3.78),     # laundry closet's louvered doors, scanned standing open in the foyer
]
# flat surfaces that clutter rests on become simple boxes (cheap, stable contacts) instead of map cells:
# name, aligned rect, top height
SURFACE_BOXES = [
    ("bed", (-5.26, 2.18, -3.88, 4.22), 0.58),          # mattress top, without the clothes and pillows on it
    ("desk_front", (-3.85, 3.86, -2.70, 4.02), 0.89),   # front strip of the desk (the monitor behind stays)
]
KEEP_SOLID = [             # aligned rects that stay solid whatever their height (walls of the closet)
    (-4.18, 1.0, -4.08, 1.6),
]


def _load_static():
    from scipy import ndimage
    H = _raw["H"].astype(np.float64).copy()
    unseen = H < 0
    outside = ndimage.binary_propagation(np.pad(np.ones_like(unseen), 1, constant_values=True) & False |
                                         np.pad(unseen, 1, constant_values=True) & False, mask=None) if False else None
    # unseen space connected to the map border is outside the home: solid. Unseen holes inside: floor.
    border = np.zeros_like(unseen)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    outside = ndimage.binary_propagation(border & unseen, mask=unseen)
    H[outside] = WALL_H
    H[unseen & ~outside] = 0.0
    H = ndimage.median_filter(H, size=3)                # scan speckle on walls and floor edges

    def idx(r):
        x0, y0, x1, y1 = r
        return (slice(max(int((y0 - _Y0) / RES), 0), int((y1 - _Y0) / RES) + 1),
                slice(max(int((x0 - _X0) / RES), 0), int((x1 - _X0) / RES) + 1))

    solid = np.zeros_like(unseen)
    for r in KEEP_SOLID:
        solid[idx(r)] = True
    H[(H < STRIP_BELOW) & ~solid] = 0.0
    lab, n = ndimage.label(H >= STRIP_BELOW)
    sizes = ndimage.sum(np.ones_like(H), lab, index=np.arange(1, n + 1)) * RES * RES
    small = np.isin(lab, np.where(sizes < STRIP_BLOB)[0] + 1) & ~solid
    H[small] = 0.0
    for r in CLEAR:
        H[idx(r)] = 0.0
    for _, r, _ in SURFACE_BOXES:
        H[idx(r)] = 0.0
    H[solid & (H < 1.0)] = WALL_H
    return H.astype(np.float32)


STATIC = _load_static()                                 # (ny, nx) heights, row 0 = southmost
_FREE = STATIC < 0.03
for _n, (_x0, _y0, _x1, _y1), _h in SURFACE_BOXES:          # the boxes are obstacles too
    _FREE[max(int((_y0 - _Y0) / RES), 0):int((_y1 - _Y0) / RES) + 1,
          max(int((_x0 - _X0) / RES), 0):int((_x1 - _X0) / RES) + 1] = False


def static_height(x, y):
    """Height of the static map at sim (x, y) (nearest cell); works on arrays too."""
    j = np.clip(np.round((np.asarray(x) + CX - _X0) / RES - 0.5).astype(int), 0, _NX - 1)
    i = np.clip(np.round((np.asarray(y) + CY - _Y0) / RES - 0.5).astype(int), 0, _NY - 1)
    return STATIC[i, j]


def _clearance():
    from scipy import ndimage
    return ndimage.distance_transform_edt(_FREE) * RES


CLEARANCE = _clearance()       # metres from each free cell to the nearest obstacle


# ----------------------------------------------------------------------------
# things the robot operates, at their measured places
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Bin:
    """An open container. ``height`` is the wall height (0 = a marked zone with no walls);
    ``z`` lifts it onto furniture."""
    name: str
    center: tuple
    inner: tuple
    height: float
    yaw: float = 0.0
    jitter: float = 0.0
    z: float = 0.0
    rgba: tuple = (0.3, 0.6, 0.65, 1)


BINS = [
    Bin("trash_can", A(1.38, -0.06), (0.14, 0.14), 0.60, rgba=(0.12, 0.12, 0.13, 1)),
    Bin("recycling", A(1.38, -0.60), (0.30, 0.22), 0.05, rgba=(0.20, 0.45, 0.80, 1)),
    Bin("laundry_basket", A(-4.42, 0.74), (0.22, 0.17), 0.32, rgba=(0.85, 0.85, 0.80, 1)),
    Bin("shoe_rack", A(-4.42, 0.22), (0.24, 0.14), 0.05, rgba=(0.35, 0.25, 0.18, 1)),
    Bin("pod_bucket", A(2.22, -3.25), (0.09, 0.09), 0.18, rgba=(0.95, 0.50, 0.15, 1)),
    Bin("iron_spot", A(-2.92, 3.94), (0.13, 0.07), 0.02, z=0.89, rgba=(0.6, 0.6, 0.6, 1)),
]

# zones with no walls, checked like containers: (name, aligned rect, z range)
ZONES = [
    ("bed_head", (-5.24, 3.68, -3.92, 4.20), (0.58, 1.0)),
]


@dataclass(frozen=True)
class Machine:
    """Front-loading machine: footprint centre, half-sizes (width, depth, height), base height, yaw
    (door faces local +y), drum opening (half-size) at ``opening_z`` above its base."""
    name: str
    center: tuple
    half: tuple = (0.30, 0.31, 0.43)
    base_z: float = 0.0
    yaw: float = math.pi / 2               # door faces west, into the foyer
    opening_half: float = 0.17
    opening_z: float = 0.48
    drum_depth: float = 0.45


# measured from the scan: stack front face at x 2.68, centred at y -3.82; washer door centre 0.5 m,
# dryer door centre 1.55 m, top of the stack about 1.85 m (24 in compact pair)
WASHER = Machine("washer", A(2.99, -3.82), base_z=0.02, opening_z=0.48, opening_half=0.17)
DRYER = Machine("dryer", A(2.99, -3.82), half=(0.30, 0.31, 0.45), base_z=0.90, opening_z=0.66, opening_half=0.16)


@dataclass(frozen=True)
class Door:
    """Hinged door. Hinge at ``hinge``; closed, the panel runs along local +x, turned by ``yaw``;
    it opens by positive rotation up to ``open_deg``. ``latch``: "lever" (locked until a lever is
    pressed) or "catch" (a magnetic catch; pull the knob)."""
    name: str
    hinge: tuple
    yaw: float
    width: float
    open_deg: float = 100.0
    height: float = 2.03
    thickness: float = 0.035
    latch: str = "lever"
    hinge_friction: float = 0.6
    handle_height: float = 0.95
    lever_length: float = 0.12
    lever_unlatch_deg: float = 30.0
    lever_spring: float = 1.5
    latch_inset: float = 0.07
    open_prob: float = 0.5               # chance it starts open


DOORS = [
    # bedroom door: hinged at the south end of the doorway, opens into the bedroom
    Door("bedroom_door", A(-2.62, 0.11), math.pi / 2, 0.85, open_deg=92, open_prob=0.85),
    # laundry closet: the louvered folding doors, simulated as one panel with a knob and a magnetic catch
    Door("laundry_door", A(2.44, -4.43), math.pi / 2, 0.94, open_deg=100, latch="catch",
         handle_height=1.0, hinge_friction=0.3, open_prob=0.3),
]
DOORWAYS = {"bedroom": (A(-2.62, 0.5), math.pi / 2), "laundry": (A(2.42, -3.95), math.pi / 2)}

DOCK_POSE = (A(0.72, -1.20), 0.0)               # foyer west wall, facing east: central, clear approach
CHAIR_HOME = A(-3.10, 3.45)                      # in front of the desk
VACUUM_DOCK = A(0.78, 1.95)                      # stick vacuum on its wall mount (static for now)


# ----------------------------------------------------------------------------
# free-space queries
# ----------------------------------------------------------------------------

def obstacle_rects(extra=()):
    """Movable or simulated obstacles as rectangles (cx, cy, hx, hy, yaw); the static map handles the rest."""
    rects = [(*b.center, b.inner[0] + 0.02, b.inner[1] + 0.02, b.yaw) for b in BINS if b.z == 0.0]
    rects.append((*WASHER.center, WASHER.half[1] + 0.02, WASHER.half[0] + 0.02, 0.0))
    rects.extend(extra)
    return rects


def _dist_to_rect(p, r):
    cx, cy, hx, hy, yaw = r
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = p[0] - cx, p[1] - cy
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return math.hypot(max(abs(lx) - hx, 0.0), max(abs(ly) - hy, 0.0))


def is_free(p, radius, rects=(), rooms=None) -> bool:
    """A circle of floor at sim ``p`` is clear of the static map, of ``rects``, and (if given)
    inside one of ``rooms`` (name -> rect)."""
    j = int((p[0] + CX - _X0) / RES)
    i = int((p[1] + CY - _Y0) / RES)
    if not (0 <= i < _NY and 0 <= j < _NX) or CLEARANCE[i, j] < radius:
        return False
    if rooms is not None and not any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for x0, y0, x1, y1 in rooms.values()):
        return False
    return all(_dist_to_rect(p, r) > radius for r in rects)


def room_of(p):
    for name, (x0, y0, x1, y1) in ROOMS.items():
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
            return name
    return None
