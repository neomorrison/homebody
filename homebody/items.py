"""The owner's actual clutter (from their scan and notes, 2026-10-19), and where each thing belongs.

    trash      the takeout bag on the bed                                   -> kitchen trash can
    recycling  two aluminium cans on the desk, two paper bags by the door,
               a cardboard box                                              -> recycling spot (or trash can)
    clothes    shirts, jeans and socks on the floor and bed                 -> laundry basket (closet) or washer
    shoes      sneakers and slides on the floor                             -> shoe rack (closet)
    pillows    stacked in the corner at the foot of the bed                  -> head of the bed
    iron       clothing iron on the floor with its cord strung out           -> its spot on the desk corner
    pods       laundry pods, in their bucket by the laundry closet          -> washer

Shapes are built to be grasped the way a person would: bags and boxes are
hollow with thin walls you can pinch, the takeout bag has a handle loop,
clothes and pillows are floppy grids of patches, the iron trails a cord made of
linked segments. Each item starts near where it was in the scan (with a little
jitter), or anywhere on free floor with ``scatter=True``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import home as H

BED, DESK = H.BED_Z, H.DESK_Z


@dataclass(frozen=True)
class Item:
    name: str
    kind: str
    shape: str                   # rigid | hollow | handled | cloth | corded
    size: tuple                  # half-sizes (rigid/hollow/handled/corded body); cloth: patch half-sizes
    mass: float
    rgba: tuple
    start: tuple                 # aligned (x, y) where it was in the scan
    on: float = 0.0              # surface height it starts on (0 floor, BED, DESK)
    geom: str = "box"            # rigid: box | cylinder
    grid: tuple = ()             # cloth: (rows, cols)
    soft: bool = False           # cloth: squishy contacts (pillows)
    stiffness: float = 0.01      # cloth: joint spring
    liquid: float = 0.0          # kg of drink when full (cans); each episode fills it 0-100%, half are empty


ITEMS = [
    Item("takeout_bag", "trash", "handled", (0.11, 0.08, 0.06), 0.45, (0.95, 0.95, 0.92, 1), (-4.15, 3.40), BED),
    Item("can_1", "recycling", "rigid", (0.033, 0.061), 0.015, (0.80, 0.10, 0.12, 1), (-3.55, 3.94), DESK, geom="cylinder", liquid=0.355),
    Item("can_2", "recycling", "rigid", (0.033, 0.061), 0.015, (0.75, 0.78, 0.82, 1), (-3.20, 3.95), DESK, geom="cylinder", liquid=0.355),
    Item("paper_bag_1", "recycling", "hollow", (0.125, 0.075, 0.15), 0.08, (0.72, 0.55, 0.35, 1), (-3.30, 1.12)),
    Item("paper_bag_2", "recycling", "hollow", (0.125, 0.075, 0.15), 0.08, (0.72, 0.55, 0.35, 1), (-3.88, 0.22)),
    Item("cardboard_box", "recycling", "hollow", (0.22, 0.17, 0.15), 0.6, (0.70, 0.55, 0.38, 1), (-3.70, 0.62)),
    Item("tshirt_1", "clothes", "cloth", (0.045, 0.04, 0.005), 0.18, (0.15, 0.15, 0.15, 1), (-3.80, 2.70), grid=(3, 3)),
    Item("tshirt_2", "clothes", "cloth", (0.045, 0.04, 0.005), 0.18, (0.20, 0.25, 0.45, 1), (-4.55, 2.85), BED, grid=(3, 3)),
    Item("jeans", "clothes", "cloth", (0.05, 0.045, 0.008), 0.60, (0.45, 0.35, 0.25, 1), (-4.30, 1.90), grid=(4, 3)),
    Item("sock_1", "clothes", "cloth", (0.04, 0.03, 0.006), 0.03, (0.10, 0.10, 0.10, 1), (-3.45, 2.05), grid=(1, 2)),
    Item("sock_2", "clothes", "cloth", (0.04, 0.03, 0.006), 0.03, (0.10, 0.10, 0.10, 1), (-3.70, 3.40), grid=(1, 2)),
    Item("sneaker_l", "shoes", "rigid", (0.14, 0.05, 0.05), 0.35, (0.95, 0.95, 0.95, 1), (-3.20, 1.82)),
    Item("sneaker_r", "shoes", "rigid", (0.14, 0.05, 0.05), 0.35, (0.95, 0.95, 0.95, 1), (-3.10, 1.95)),
    Item("slide_l", "shoes", "rigid", (0.13, 0.045, 0.02), 0.15, (0.08, 0.08, 0.08, 1), (-3.22, 2.45)),
    Item("slide_r", "shoes", "rigid", (0.13, 0.045, 0.02), 0.15, (0.08, 0.08, 0.08, 1), (-3.10, 2.38)),
    Item("pillow_1", "pillow", "cloth", (0.0625, 0.044, 0.05), 0.5, (0.10, 0.10, 0.12, 1), (-4.90, 1.86), grid=(4, 4),
         soft=True, stiffness=0.3),
    Item("pillow_2", "pillow", "cloth", (0.0625, 0.044, 0.05), 0.5, (0.12, 0.12, 0.14, 1), (-4.90, 1.86), 0.11, grid=(4, 4),
         soft=True, stiffness=0.3),
    Item("iron", "iron", "corded", (0.12, 0.055, 0.06), 1.2, (0.55, 0.65, 0.85, 1), (-3.80, 1.35)),
] + [Item(f"pod_{k}", "pod", "rigid", (0.02, 0.018, 0.011), 0.02, (0.20, 0.50, 0.95, 1), (0, 0)) for k in range(1, 5)]
ITEM_NAMES = [i.name for i in ITEMS]
GOALS = {
    "trash": ("trash_can",), "recycling": ("recycling", "trash_can"), "clothes": ("laundry_basket", "washer"),
    "shoes": ("shoe_rack",), "pillow": ("bed_head",), "iron": ("iron_spot",), "pod": ("washer",),
}
KINDS = list(GOALS)
PARK = (40.0, 0.0, -0.9)
SURF = 'priority="2" condim="4"'


def _f(*v):
    return " ".join(f"{x:.5g}" for x in v)


def _rigid(it, name, head):
    hs = it.size
    g = (f'type="cylinder" size="{_f(*hs)}"' if it.geom == "cylinder" else f'type="box" size="{_f(*hs)}"')
    return (f'{head}<geom name="{name}" {g} mass="{it.mass}" {SURF} friction="0.6 0.01 0.001" '
            f'rgba="{_f(*it.rgba)}"/></body>')


def _hollow(it, name, head):
    """Open-topped thin-walled container (paper bag, cardboard box): pinch a wall to carry it."""
    hx, hy, hz = it.size
    t = 0.003
    m = it.mass / 5
    s = f'{SURF} friction="0.7 0.01 0.001" rgba="{_f(*it.rgba)}"'
    walls = [
        f'<geom name="{name}" type="box" size="{_f(hx, hy, t)}" pos="0 0 {-hz + t:.4f}" mass="{m:.4f}" {s}/>',
        f'<geom type="box" size="{_f(t, hy, hz)}" pos="{_f(hx - t, 0, 0)}" mass="{m:.4f}" {s}/>',
        f'<geom type="box" size="{_f(t, hy, hz)}" pos="{_f(-hx + t, 0, 0)}" mass="{m:.4f}" {s}/>',
        f'<geom type="box" size="{_f(hx, t, hz)}" pos="{_f(0, hy - t, 0)}" mass="{m:.4f}" {s}/>',
        f'<geom type="box" size="{_f(hx, t, hz)}" pos="{_f(0, -hy + t, 0)}" mass="{m:.4f}" {s}/>',
    ]
    return head + "".join(walls) + "</body>"


def _handled(it, name, head):
    """A bag of food containers with a handle loop on top."""
    hx, hy, hz = it.size
    s = f'{SURF} friction="0.6 0.01 0.001" rgba="{_f(*it.rgba)}"'
    return (f'{head}<geom name="{name}" type="box" size="{_f(hx, hy, hz)}" mass="{it.mass - 0.01:.3f}" {s}/>'
            f'<geom type="capsule" fromto="{_f(-0.05, 0, hz, 0, 0, hz + 0.07)}" size="0.004" mass="0.005" {s}/>'
            f'<geom type="capsule" fromto="{_f(0, 0, hz + 0.07, 0.05, 0, hz)}" size="0.004" mass="0.005" {s}/></body>')


def _corded(it, name, head, links=9, link_len=0.16):
    """Clothing iron with its cord: a chain of capsules on ball joints, ending in a plug."""
    hx, hy, hz = it.size
    s = f'{SURF} friction="0.6 0.01 0.001"'
    body = (f'{head}<geom name="{name}" type="box" size="{_f(hx, hy, hz)}" mass="{it.mass:.3f}" {s} rgba="{_f(*it.rgba)}"/>'
            f'<geom type="box" size="{_f(hx * 0.6, 0.012, 0.03)}" pos="{_f(-0.01, 0, hz + 0.03)}" mass="0.05" {s} rgba="0.2 0.2 0.25 1"/>')
    chain, close = "", ""
    x = -hx
    for k in range(links):
        pos = _f(x, 0, -hz + 0.01) if k == 0 else _f(-link_len, 0, 0)
        chain += (f'<body name="{name}_cord_{k}" pos="{pos}">'
                  f'<joint name="{name}_cord_{k}" type="ball" range="0 120" stiffness="0.002" damping="0.002" armature="0.0002"/>'
                  f'<geom type="capsule" fromto="0 0 0 {-link_len:.3f} 0 0" size="0.0045" mass="0.012" {s} rgba="0.95 0.95 0.95 1"/>')
        close += "</body>"
        x = 0
    plug = f'<geom name="{name}_plug" type="box" size="0.02 0.012 0.008" pos="{_f(-link_len - 0.02, 0, 0)}" mass="0.03" {s} rgba="0.9 0.9 0.9 1"/>'
    return body + chain + plug + close + "</body>"


def _cloth(it, name, head):
    """A grid of patches as a tree rooted at the centre patch, children on ball joints."""
    rows, cols = it.grid
    hx, hy, hz = it.size
    m = it.mass / (rows * cols)
    r0, c0 = rows // 2, cols // 2
    soft = ' solref="0.06 1" solimp="0.6 0.95 0.02"' if it.soft else ""
    surf = f'{SURF} friction="0.9 0.01 0.001" rgba="{_f(*it.rgba)}"{soft}'
    rng = "40" if it.soft else "100"

    def body(r, c, parent):
        bname = name if parent is None else f"{name}_{r}{c}"
        if parent is None:
            h = head
        else:
            pr, pc = parent
            dx, dy = (r - pr) * 2 * hx, (c - pc) * 2 * hy
            h = (f'<body name="{bname}" pos="{_f(dx, dy, 0)}">'
                 f'<joint name="{bname}" type="ball" pos="{_f(-dx / 2, -dy / 2, 0)}" range="0 {rng}" '
                 f'stiffness="{it.stiffness}" damping="{it.stiffness * 0.3 + 0.003:.4f}" armature="0.0001"/>')
        geom = f'<geom name="{bname}" type="box" size="{_f(hx * 0.98, hy * 0.98, hz)}" mass="{m:.4f}" {surf}/>'
        kids = []
        if c == c0:
            for nr in (r - 1, r + 1):
                if 0 <= nr < rows and abs(nr - r0) > abs(r - r0):
                    kids.append(body(nr, c, (r, c)))
        for nc in (c - 1, c + 1):
            if 0 <= nc < cols and abs(nc - c0) > abs(c - c0):
                kids.append(body(r, nc, (r, c)))
        return h + geom + "".join(kids) + "</body>"
    return body(r0, c0, None)


BUILD = {"rigid": _rigid, "hollow": _hollow, "handled": _handled, "corded": _corded, "cloth": _cloth}


def items_xml() -> str:
    """Free bodies. Must come after the robot so the robot's root comes first among free joints."""
    out = []
    for k, it in enumerate(ITEMS):
        name = f"item_{it.name}"
        head = f'<body name="{name}" pos="{PARK[0] + 0.5 * k} {PARK[1]} {PARK[2]}"><freejoint name="{name}"/>'
        out.append(BUILD[it.shape](it, name, head))
    return "\n    ".join(out)


def _footprint(it):
    """Radius of the circle an item covers lying on the floor."""
    if it.shape == "cloth":
        rows, cols = it.grid
        return 0.5 * math.hypot(rows * 2 * it.size[0], cols * 2 * it.size[1])
    if it.shape == "corded":
        return math.hypot(it.size[0], it.size[1]) + 0.1
    return math.hypot(it.size[0], it.size[1])


def _resting_half_height(it):
    if it.shape == "cloth":
        return it.size[2]
    if it.geom == "cylinder":
        return it.size[1]
    return it.size[2]


class Items:
    def __init__(self, model, container_sites: dict):
        m = model
        self.m = m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.adr = [(m.jnt_qposadr[jid(f"item_{n}")], m.jnt_dofadr[jid(f"item_{n}")]) for n in ITEM_NAMES]
        self.body = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"item_{n}") for n in ITEM_NAMES]
        self.sites = container_sites
        self.active = np.zeros(len(ITEMS), bool)
        self.mass0 = m.body_mass[self.body].copy()
        self.inertia0 = m.body_inertia[self.body].copy()
        self.fill = np.zeros(len(ITEMS))                # fraction full, for items that can hold liquid
        self.spilled = np.zeros(len(ITEMS), bool)

    def set_fill(self, k, fill):
        """Fill item ``k`` with liquid (0..1 of ITEMS[k].liquid kg); mass and inertia follow."""
        it = ITEMS[k]
        self.fill[k] = fill
        mass = self.mass0[k] + fill * it.liquid
        self.m.body_mass[self.body[k]] = mass
        self.m.body_inertia[self.body[k]] = self.inertia0[k] * mass / self.mass0[k]
        self._dirty = True

    def apply(self, data):
        """Recompute MuJoCo's derived constants after mass changes (call before stepping)."""
        if getattr(self, "_dirty", False):
            saved = data.qpos.copy(), data.qvel.copy(), data.act.copy(), data.time
            mujoco.mj_setConst(self.m, data)             # uses data as scratch at qpos0: restore after
            data.qpos[:], data.qvel[:], data.act[:], data.time = saved
            mujoco.mj_forward(self.m, data)
            self._dirty = False

    def update_spills(self, data, tilt_cos=0.5):
        """A container with liquid tipped past 60 degrees has spilled: it's empty now, and the floor
        needs cleaning. Returns the item indices that spilled this call."""
        out = []
        for k, it in enumerate(ITEMS):
            if it.liquid and self.active[k] and self.fill[k] > 0.05 and data.xmat[self.body[k]][8] < tilt_cos:
                self.spilled[k] = True
                self.set_fill(k, 0.0)
                out.append(k)
        self.apply(data)
        return out

    def place(self, rng, kinds, qpos, qvel, height_at, is_free, bucket_pos, scatter=False, jitter=0.08,
              dropout=0.0):
        """Put the items of ``kinds`` where they were in the scan (plus jitter), or anywhere free
        with ``scatter``; ``dropout`` leaves each one out with that probability. Pods go in the bucket."""
        self.active[:] = False
        self.spilled[:] = False
        for k, it in enumerate(ITEMS):
            if it.liquid:
                self.set_fill(k, 0.0 if rng.uniform() < 0.5 else rng.uniform(0.2, 1.0))
        npod = 0
        placed = []                                     # floor footprints so far: ((x, y), radius)
        for k, it in enumerate(ITEMS):
            qa, va = self.adr[k]
            qvel[va:va + 6] = 0
            park = [PARK[0] + 0.5 * k, PARK[1], PARK[2], 1, 0, 0, 0]
            if it.kind not in kinds or rng.uniform() < dropout:
                qpos[qa:qa + 7] = park
                continue
            yaw = rng.uniform(-math.pi, math.pi)
            if it.kind == "pod":
                gx, gy = ((-0.04, -0.04), (0.04, -0.04), (-0.04, 0.04), (0.04, 0.04))[npod % 4]
                pos = (bucket_pos[0] + gx, bucket_pos[1] + gy, bucket_pos[2] + 0.02 + 0.03 * (npod // 4))
                npod += 1
            elif scatter and it.on == 0.0:
                pos = None
                for _ in range(80):
                    p = np.array([rng.uniform(-H.EXTENT[0] / 2, H.EXTENT[0] / 2), rng.uniform(-H.EXTENT[1] / 2, H.EXTENT[1] / 2)])
                    if H.room_of(p) in ("bedroom", "hallway", "living") and is_free(p, 0.2):
                        pos = (p[0], p[1], height_at(*p) + _resting_half_height(it) + 0.015)
                        break
                if pos is None:
                    qpos[qa:qa + 7] = park
                    continue
            else:
                x, y = H.A(*it.start)
                if it.on == 0.0:
                    r = _footprint(it)
                    spot, jit = None, jitter
                    for _ in range(40):                 # jitter, but keep clear of walls, furniture and other items
                        px, py = x + rng.uniform(-jit, jit), y + rng.uniform(-jit, jit)
                        if is_free((px, py), r) and all(math.hypot(px - qx, py - qy) > r + rq for (qx, qy), rq in placed):
                            spot = (px, py)
                            break
                        jit = min(jit * 1.2, 0.5)
                    if spot is None:                    # no room: leave it out rather than overlap something
                        qpos[qa:qa + 7] = park
                        continue
                    x, y = spot
                    placed.append((spot, r))
                    z = height_at(x, y) + _resting_half_height(it) + 0.015
                else:
                    x, y = x + rng.uniform(-0.03, 0.03), y + rng.uniform(-0.03, 0.03)
                    z = it.on + _resting_half_height(it) + 0.015
                    if it.name == "pillow_2":
                        z = 0.115 + _resting_half_height(it) + 0.02     # on top of pillow_1
                pos = (x, y, z)
            if it.shape == "cloth" and it.kind == "pillow":
                yaw = rng.uniform(-0.3, 0.3)
            qpos[qa:qa + 7] = [*pos, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            self.active[k] = True

    def locate(self, data) -> list:
        """For each item: the container or zone it rests in (name), or None."""
        boxes = {name: (data.site_xpos[s], data.site_xmat[s].reshape(3, 3), self.m.site_size[s])
                 for name, s in self.sites.items()}
        out = []
        for k, b in enumerate(self.body):
            where = None
            if self.active[k]:
                p = data.xpos[b]
                for name, (c, R, hs) in boxes.items():
                    if np.all(np.abs(R.T @ (p - c)) <= hs):
                        where = name
                        break
            out.append(where)
        return out

    def tidy_count(self, data):
        where = self.locate(data)
        done, detail = 0, {}
        for k, it in enumerate(ITEMS):
            if not self.active[k]:
                continue
            done += where[k] in GOALS[it.kind]
            detail[it.name] = where[k]
        return done, detail
