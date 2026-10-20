"""The owner's home, recreated from their phone LiDAR scan (2026-10-19).

The scan is the measuring tape, not the model. Every wall and piece of
furniture here is a clean object placed and sized from measurements of the
scan (scripts/import_scan.py, then reading heights and extents off the height
map), and the floor and rug use textures cut from the scan's photos. So the
sim has:

* straight walls generated around the rooms (doorways fall out of the room
  layout), and interior partitions;
* furniture as separate parts with their real heights, and space underneath
  where there is some (desk, tables);
* none of the scan's clutter baked in: the owner's mess is in ``items`` as
  movable objects, so cameras see only what is really there in the sim;
* colours and small furniture positions that are randomised per episode
  (``env``), so camera-based skills learn the room, not one exact render.

Coordinates: metres, sim frame (x east, y north, origin at the centre of the
scan). Everything is written in the scan's aligned frame with ``A()`` /
``AR()``, so it can be checked against the scan's top-down render.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

_SCAN = os.path.join(os.path.dirname(__file__), "assets", "home_scan.npz")
_raw = np.load(_SCAN)
_X0, _Y0 = float(_raw["x0"]), float(_raw["y0"])
_NY, _NX = _raw["H"].shape
_SRES = float(_raw["res"])
CX, CY = _X0 + _NX * _SRES / 2, _Y0 + _NY * _SRES / 2   # aligned-frame centre = sim origin
EXTENT = (_NX * _SRES, _NY * _SRES)
SCAN_ROT = float(_raw["rot"])
SCAN_FLOOR = float(_raw["floor"])
SCAN_SOURCE = str(_raw["source"])
TEXTURES = os.path.join(os.path.dirname(__file__), "assets", "textures")


def A(x, y):
    """Aligned scan frame -> sim frame."""
    return (x - CX, y - CY)


def AR(x0, y0, x1, y1):
    """Aligned rectangle -> sim rectangle (x0, y0, x1, y1)."""
    a, b = A(x0, y0), A(x1, y1)
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


# ----------------------------------------------------------------------------
# floor plan (aligned rects, measured from the scan's floor extents)
# ----------------------------------------------------------------------------

FLOOR_RECTS = {
    "bedroom": (-5.30, 0.00, -2.58, 4.25),       # includes the walk-in closet in its south-west corner
    "hallway": (-2.58, 0.00, 0.64, 1.00),
    "living": (0.48, 0.90, 4.15, 4.30),
    "kitchen": (0.62, -2.05, 4.15, 1.00),
    "foyer": (0.62, -3.30, 3.35, -1.95),
    "corridor": (-2.60, -4.30, 2.45, -3.22),      # south corridor, runs west to the rest of the flat
    "laundry": (2.45, -4.40, 3.33, -3.45),         # closet: stacked washer and dryer
    "back_room": (1.35, -7.00, 3.35, -4.25),
}
# the laundry closet's south wall separates it from the back room
ROOMS = {k: AR(*v) for k, v in FLOOR_RECTS.items()}
ROOMS["closet"] = AR(-5.30, 0.00, -4.20, 1.47)
WALL_T = 0.10
WALL_H = 2.40
# interior partitions inside the floor rects: aligned (x0, y0, x1, y1) of the wall itself
PARTITIONS = [
    (-5.30, 1.47, -4.15, 1.60),                    # closet's north wall (the bed's foot is beyond it)
    (-4.20, 0.95, -4.10, 1.60),                    # closet's east wall, beside its open side
    (2.45, -3.45, 3.33, -3.30),                    # laundry closet's north wall
    (2.45, -4.45, 3.35, -4.33),                    # laundry closet's south wall (back room beyond)
    (-2.60, -4.30, -2.50, -3.22),                  # end of the scanned part of the south corridor
]


@dataclass(frozen=True)
class Part:
    """A static piece of furniture: aligned rect footprint, z range, material. ``shape`` box or cylinder
    (a cylinder fits inside the rect). ``group`` names the piece it belongs to, for randomisation."""
    name: str
    rect: tuple
    z0: float
    z1: float
    material: str
    shape: str = "box"
    group: str = ""


def _legs(name, rect, top_z, mat, inset=0.04, r=0.025):
    x0, y0, x1, y1 = rect
    out = []
    for k, (x, y) in enumerate(((x0 + inset, y0 + inset), (x1 - inset, y0 + inset),
                                (x0 + inset, y1 - inset), (x1 - inset, y1 - inset))):
        out.append(Part(f"{name}_leg{k}", (x - r, y - r, x + r, y + r), 0.0, top_z, mat, group=name))
    return out


FURNITURE = [
    # bedroom
    Part("bed_frame", (-5.26, 2.18, -3.88, 4.12), 0.0, 0.30, "wood_light", group="bed"),
    Part("mattress", (-5.24, 2.20, -3.90, 4.10), 0.30, 0.58, "mattress", group="bed"),
    Part("headboard", (-5.26, 4.12, -3.88, 4.22), 0.0, 1.04, "wood_light", group="bed"),
    Part("desk_top", (-3.85, 3.84, -2.70, 4.22), 0.77, 0.80, "white", group="desk"),
    *_legs("desk", (-3.85, 3.84, -2.70, 4.22), 0.77, "white"),
    Part("monitor", (-3.45, 4.14, -2.95, 4.18), 0.86, 1.22, "black", group="desk"),
    Part("monitor_foot", (-3.25, 4.08, -3.15, 4.20), 0.80, 0.86, "black", group="desk"),
    Part("closet_clothes", (-5.28, 0.97, -4.22, 1.45), 0.85, 1.75, "clothes", group="closet"),
    Part("closet_shelf", (-5.28, 0.97, -4.22, 1.47), 1.75, 1.78, "white", group="closet"),
    Part("suitcase", (-3.00, 1.25, -2.65, 2.05), 0.0, 0.45, "suitcase"),
    Part("mirror", (-2.66, 2.40, -2.61, 3.10), 0.05, 1.60, "mirror"),
    # living room
    Part("sofa_seat", (3.15, 1.85, 4.05, 4.25), 0.0, 0.41, "sofa", group="sofa"),
    Part("sofa_back", (3.88, 1.85, 4.13, 4.25), 0.0, 0.74, "sofa", group="sofa"),
    Part("sofa_chaise", (2.55, 3.62, 3.15, 4.25), 0.0, 0.39, "sofa", group="sofa"),
    Part("sofa_arm", (3.15, 1.80, 3.88, 1.98), 0.0, 0.58, "sofa", group="sofa"),
    Part("bean_bag", (1.70, 3.95, 2.40, 4.30), 0.0, 0.47, "bean_bag", "cylinder"),
    Part("coffee_table", (1.85, 3.30, 2.45, 3.90), 0.42, 0.45, "glass", "cylinder", group="coffee_table"),
    Part("coffee_table_base", (2.12, 3.57, 2.18, 3.63), 0.0, 0.42, "black", "cylinder", group="coffee_table"),
    Part("tv_stand", (0.52, 2.00, 0.92, 3.80), 0.0, 0.50, "wood_dark", group="tv"),
    Part("tv", (0.55, 2.30, 0.62, 3.50), 0.60, 1.30, "black", group="tv"),
    Part("vacuum", (0.55, 1.55, 0.70, 1.72), 0.0, 1.15, "vacuum"),
    # kitchen
    Part("island", (1.62, -0.20, 3.00, 0.50), 0.0, 0.90, "cabinet", group="island"),
    Part("island_top", (1.58, -0.24, 3.04, 0.54), 0.90, 0.93, "counter", group="island"),
    Part("counter", (3.45, -1.60, 4.15, 0.55), 0.0, 0.90, "cabinet", group="kitchen"),
    Part("counter_top", (3.42, -1.60, 4.15, 0.58), 0.90, 0.93, "counter", group="kitchen"),
    Part("upper_cabinets", (3.80, -1.60, 4.15, 0.55), 1.45, 2.10, "cabinet", group="kitchen"),
    Part("fridge", (3.40, -2.30, 4.15, -1.62), 0.0, 1.80, "steel", group="kitchen"),
    Part("stool_1", (1.90, 0.62, 2.20, 0.92), 0.0, 0.65, "black", "cylinder"),
    Part("stool_2", (2.40, 0.62, 2.70, 0.92), 0.0, 0.65, "black", "cylinder"),
]
MOVABLE = {"bean_bag": 0.10, "coffee_table": 0.08, "stool_1": 0.10, "stool_2": 0.10}   # jitter (m) per reset

# base colours (from the scan's photos where the photo was usable; the scan reads dark, so brightened)
MATERIALS = {
    "wall": (0.88, 0.86, 0.80, 1), "wood_light": (0.86, 0.76, 0.62, 1), "wood_dark": (0.40, 0.30, 0.22, 1),
    "mattress": (0.93, 0.89, 0.82, 1), "white": (0.92, 0.92, 0.93, 1), "black": (0.10, 0.10, 0.11, 1),
    "clothes": (0.22, 0.24, 0.30, 1), "suitcase": (0.55, 0.12, 0.15, 1), "mirror": (0.75, 0.80, 0.85, 1),
    "sofa": (0.62, 0.56, 0.50, 1), "bean_bag": (0.85, 0.85, 0.82, 1), "glass": (0.70, 0.78, 0.80, 0.5),
    "cabinet": (0.93, 0.93, 0.92, 1), "counter": (0.25, 0.25, 0.26, 1), "steel": (0.72, 0.73, 0.74, 1),
    "vacuum": (0.45, 0.30, 0.60, 1),
}


# ----------------------------------------------------------------------------
# walls: generated around the union of the floor rects
# ----------------------------------------------------------------------------

_GRID = 0.02


def _union_grid():
    xs = np.arange(-EXTENT[0] / 2, EXTENT[0] / 2, _GRID)
    ys = np.arange(-EXTENT[1] / 2, EXTENT[1] / 2, _GRID)
    X, Y = np.meshgrid(xs + _GRID / 2, ys + _GRID / 2)
    g = np.zeros(X.shape, bool)
    for x0, y0, x1, y1 in (AR(*r) for r in FLOOR_RECTS.values()):
        g |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    return g


def wall_boxes():
    """Boxes (cx, cy, hx, hy): a WALL_T shell around the union of the floor rects (corners closed, nothing
    poking into doorways), split into rectangles; plus the interior partitions."""
    from scipy import ndimage
    g = _union_grid()
    n = int(round(WALL_T / _GRID))
    shell = ndimage.binary_dilation(g, structure=np.ones((3, 3), bool), iterations=n) & ~g
    ox, oy = -EXTENT[0] / 2, -EXTENT[1] / 2
    boxes, open_runs = [], {}                         # (j0, j1) -> first row of a run stack
    rows = shell.shape[0]
    for i in range(rows + 1):
        runs = set()
        if i < rows:
            r = shell[i]
            j = 0
            while j < r.size:
                if r[j]:
                    k = j
                    while k < r.size and r[k]:
                        k += 1
                    runs.add((j, k))
                    j = k
                else:
                    j += 1
        for key in list(open_runs):                   # close stacks that do not continue
            if key not in runs:
                i0 = open_runs.pop(key)
                j0, j1 = key
                x0, x1, y0, y1 = ox + j0 * _GRID, ox + j1 * _GRID, oy + i0 * _GRID, oy + i * _GRID
                boxes.append(((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2))
        for key in runs:
            open_runs.setdefault(key, i)
    for r in PARTITIONS:
        x0, y0, x1, y1 = AR(*r)
        boxes.append(((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2))
    return boxes


WALLS = wall_boxes()


# ----------------------------------------------------------------------------
# things the robot operates, at their measured places
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Bin:
    """An open container. ``height`` is the wall height (under 3 cm = a marked spot with no walls);
    ``z`` lifts it onto furniture."""
    name: str
    center: tuple
    inner: tuple
    height: float
    yaw: float = 0.0
    jitter: float = 0.0
    z: float = 0.0
    rgba: tuple = (0.3, 0.6, 0.65, 1)


DESK_Z = 0.80
BED_Z = 0.58
BINS = [
    Bin("trash_can", A(1.38, -0.06), (0.14, 0.14), 0.60, rgba=(0.12, 0.12, 0.13, 1)),
    Bin("recycling", A(1.38, -0.60), (0.30, 0.22), 0.02, rgba=(0.20, 0.45, 0.80, 1)),
    Bin("laundry_basket", A(-4.62, 0.74), (0.22, 0.17), 0.32, rgba=(0.85, 0.85, 0.80, 1)),
    Bin("shoe_rack", A(-4.62, 0.22), (0.24, 0.14), 0.02, rgba=(0.35, 0.25, 0.18, 1)),
    Bin("pod_bucket", A(2.22, -3.40), (0.09, 0.09), 0.18, rgba=(0.95, 0.50, 0.15, 1)),
    Bin("iron_spot", A(-2.92, 3.94), (0.13, 0.07), 0.02, z=DESK_Z, rgba=(0.6, 0.6, 0.6, 1)),
]
ZONES = [("bed_head", (-5.24, 3.68, -3.92, 4.10), (BED_Z, 1.0))]


@dataclass(frozen=True)
class Machine:
    """Front-loading machine: footprint centre, half-sizes (width, depth, height), base height, yaw
    (door faces local +y), drum opening (half-size) at ``opening_z`` above its base."""
    name: str
    center: tuple
    half: tuple = (0.30, 0.31, 0.43)
    base_z: float = 0.0
    yaw: float = math.pi / 2
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
    open_prob: float = 0.5


DOORS = [
    Door("bedroom_door", A(-2.58, 0.08), math.pi / 2, 0.86, open_deg=92, open_prob=0.85),
    Door("laundry_door", A(2.42, -4.37), math.pi / 2, 0.88, open_deg=100, latch="catch",
         handle_height=1.0, hinge_friction=0.3, open_prob=0.3),
]
DOORWAYS = {"bedroom": (A(-2.58, 0.5), math.pi / 2), "laundry": (A(2.42, -3.95), math.pi / 2)}

DOCK_POSE = (A(0.62, -1.20), 0.0)               # foyer west wall, facing east: central, clear approach
CHAIR_HOME = A(-3.20, 3.45)                     # in front of the desk


# ----------------------------------------------------------------------------
# free-space queries (rasterised from the walls and furniture)
# ----------------------------------------------------------------------------

RES = 0.04


def _raster(block_below):
    """True where the floor is free of walls and of furniture that comes lower than ``block_below`` m."""
    xs = np.arange(-EXTENT[0] / 2, EXTENT[0] / 2, RES) + RES / 2
    ys = np.arange(-EXTENT[1] / 2, EXTENT[1] / 2, RES) + RES / 2
    X, Y = np.meshgrid(xs, ys)
    free = np.zeros(X.shape, bool)
    for name, (x0, y0, x1, y1) in ROOMS.items():
        if name != "closet":
            free |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    for cx, cy, hx, hy in WALLS:
        free &= ~((np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy))
    for p in FURNITURE:
        if p.z0 < block_below:
            x0, y0, x1, y1 = AR(*p.rect)
            free &= ~((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))
    x, y = WASHER.center
    free &= ~((np.abs(X - x) <= WASHER.half[1]) & (np.abs(Y - y) <= WASHER.half[0]))
    return free


def _clearance(free):
    from scipy import ndimage
    return ndimage.distance_transform_edt(free) * RES


ROBOT_FREE = _raster(1.95)          # the robot is 1.9 m tall: anything it would drive under is in the way
ITEM_FREE = _raster(0.05)           # loose items can lie under the desk and tables
CLEARANCE = _clearance(ROBOT_FREE)
ITEM_CLEARANCE = _clearance(ITEM_FREE)


def obstacle_rects(extra=()):
    """Movable or simulated obstacles as rectangles (cx, cy, hx, hy, yaw); walls and furniture are in the raster."""
    rects = [(*b.center, b.inner[0] + 0.02, b.inner[1] + 0.02, b.yaw) for b in BINS if b.z == 0.0]
    rects.extend(extra)
    return rects


def _dist_to_rect(p, r):
    cx, cy, hx, hy, yaw = r
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = p[0] - cx, p[1] - cy
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return math.hypot(max(abs(lx) - hx, 0.0), max(abs(ly) - hy, 0.0))


def is_free(p, radius, rects=(), rooms=None, for_items=False) -> bool:
    """A circle of floor at sim ``p`` is clear of walls, furniture and ``rects``, and (if given) inside one of
    ``rooms``. ``for_items`` lets it sit under desks and tables."""
    cl = ITEM_CLEARANCE if for_items else CLEARANCE
    j = int((p[0] + EXTENT[0] / 2) / RES)
    i = int((p[1] + EXTENT[1] / 2) / RES)
    if not (0 <= i < cl.shape[0] and 0 <= j < cl.shape[1]) or cl[i, j] < radius:
        return False
    if rooms is not None and not any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for x0, y0, x1, y1 in rooms.values()):
        return False
    return all(_dist_to_rect(p, r) > radius for r in rects)


def room_of(p):
    for name in ("closet", *FLOOR_RECTS):
        x0, y0, x1, y1 = ROOMS[name]
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
            return name
    return None
