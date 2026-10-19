"""Things to tidy, and where each kind belongs.

Every item is a loose body the robot can push, knock over, pick up and drop.
Sizes and masses are real household objects. Clothes are floppy: a grid of
fabric patches joined by weak spring joints, so a pinch lifts one patch and the
rest hangs and drapes (a cheap stand-in for full cloth simulation).

A container counts an item when the item's centre is inside the container's
inner volume. Goals: trash -> trash can; clothes -> laundry basket or washer;
pods -> washer. The robot's deck tote is for carrying and counts for nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from . import home as H


@dataclass(frozen=True)
class Item:
    name: str
    kind: str          # trash | clothes | pod
    geom: str          # rigid items: MJCF geom attributes; cloth: "cloth"
    mass: float
    rgba: tuple
    where: str         # floor | bucket (where it starts)
    grid: tuple = ()   # cloth: (rows, cols) of patches
    patch: tuple = ()  # cloth: patch half-sizes (x, y, z)


ITEMS = [
    Item("paper_ball_1", "trash", 'type="sphere" size="0.035"', 0.01, (0.95, 0.95, 0.92, 1), "floor"),
    Item("paper_ball_2", "trash", 'type="sphere" size="0.03"', 0.01, (0.95, 0.95, 0.92, 1), "floor"),
    Item("soda_can", "trash", 'type="cylinder" size="0.033 0.06"', 0.015, (0.80, 0.10, 0.10, 1), "floor"),
    Item("bottle", "trash", 'type="cylinder" size="0.035 0.1"', 0.03, (0.70, 0.85, 0.95, 0.8), "floor"),
    Item("wrapper", "trash", 'type="box" size="0.06 0.04 0.004"', 0.005, (0.90, 0.70, 0.10, 1), "floor"),
    Item("tshirt_1", "clothes", "cloth", 0.18, (0.20, 0.30, 0.70, 1), "floor", (3, 3), (0.045, 0.04, 0.005)),
    Item("tshirt_2", "clothes", "cloth", 0.18, (0.15, 0.15, 0.15, 1), "floor", (3, 3), (0.045, 0.04, 0.005)),
    Item("sock_1", "clothes", "cloth", 0.03, (0.95, 0.95, 0.95, 1), "floor", (1, 2), (0.04, 0.03, 0.006)),
    Item("sock_2", "clothes", "cloth", 0.03, (0.95, 0.95, 0.95, 1), "floor", (1, 2), (0.04, 0.03, 0.006)),
    Item("jeans", "clothes", "cloth", 0.60, (0.20, 0.30, 0.50, 1), "floor", (4, 3), (0.05, 0.045, 0.008)),
] + [Item(f"pod_{k}", "pod", 'type="box" size="0.02 0.018 0.011"', 0.02, (0.20, 0.50, 0.95, 1), "bucket")
     for k in range(1, 5)]
ITEM_NAMES = [i.name for i in ITEMS]
GOALS = {"trash": ("trash_can",), "clothes": ("laundry_basket", "washer"), "pod": ("washer",)}
PARK = (30.0, 0.0, -0.9)


def _f(*v):
    return " ".join(f"{x:.5g}" for x in v)


def _cloth_xml(it: Item, k: int) -> str:
    """A grid of patches as a tree rooted at the centre patch (free joint), children on ball joints."""
    rows, cols = it.grid
    hx, hy, hz = it.patch
    m = it.mass / (rows * cols)
    r0, c0 = rows // 2, cols // 2
    surf = f'priority="2" condim="4" friction="0.9 0.01 0.001" rgba="{_f(*it.rgba)}"'

    def body(r, c, parent):
        name = f"item_{it.name}" if parent is None else f"item_{it.name}_{r}{c}"
        if parent is None:
            head = f'<body name="{name}" pos="{PARK[0] + 0.3 * k} {PARK[1]} {PARK[2]}"><freejoint name="{name}"/>'
        else:
            pr, pc = parent
            dx, dy = (r - pr) * 2 * hx, (c - pc) * 2 * hy
            head = (f'<body name="{name}" pos="{_f(dx, dy, 0)}">'
                    f'<joint name="{name}" type="ball" pos="{_f(-dx / 2, -dy / 2, 0)}" range="0 100" '
                    f'stiffness="0.01" damping="0.004" armature="0.0001"/>')
        geom = f'<geom name="{name}" type="box" size="{_f(hx * 0.98, hy * 0.98, hz)}" mass="{m:.4f}" {surf}/>'
        kids = []
        if c == c0:                 # spine runs along the rows; branches run along the columns
            for nr in (r - 1, r + 1):
                if 0 <= nr < rows and abs(nr - r0) > abs(r - r0):
                    kids.append(body(nr, c, (r, c)))
        for nc in (c - 1, c + 1):
            if 0 <= nc < cols and abs(nc - c0) > abs(c - c0):
                kids.append(body(r, nc, (r, c)))
        return head + geom + "".join(kids) + "</body>"
    return body(r0, c0, None)


def items_xml() -> str:
    """Free bodies. Must come after the robot so the robot's root comes first among free joints."""
    out = []
    for k, it in enumerate(ITEMS):
        if it.geom == "cloth":
            out.append(_cloth_xml(it, k))
        else:
            out.append(
                f'<body name="item_{it.name}" pos="{PARK[0] + 0.3 * k} {PARK[1]} {PARK[2]}"><freejoint name="item_{it.name}"/>'
                f'<geom name="item_{it.name}" {it.geom} mass="{it.mass}" priority="2" condim="4" friction="0.6 0.01 0.001" '
                f'rgba="{_f(*it.rgba)}"/></body>')
    return "\n    ".join(out)


class Items:
    def __init__(self, model, container_sites: dict):
        """``container_sites``: name -> site id of a box site spanning the container's inner volume."""
        m = model
        self.m = m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.adr = [(m.jnt_qposadr[jid(f"item_{n}")], m.jnt_dofadr[jid(f"item_{n}")]) for n in ITEM_NAMES]
        self.body = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"item_{n}") for n in ITEM_NAMES]
        self.sites = container_sites
        self.active = np.zeros(len(ITEMS), bool)

    def place(self, rng, n_trash, n_clothes, n_pods, is_free, qpos, qvel, height_at, bucket_site_pos, tops=()):
        """Scatter trash and clothes on free floor (some on furniture ``tops`` [(xy, half, top_z)]),
        and pods in the bucket. Unused items are parked out of the house."""
        want = {"trash": n_trash, "clothes": n_clothes, "pod": n_pods}
        self.active[:] = False
        placed = []
        for k, it in enumerate(ITEMS):
            qa, va = self.adr[k]
            qvel[va:va + 6] = 0
            yaw = rng.uniform(-math.pi, math.pi)
            quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            pos = None
            if want[it.kind] > 0:
                if it.where == "bucket":       # pods sit side by side on the bucket floor, then a second layer
                    c = bucket_site_pos
                    n = int(sum(self.active[[j for j, i in enumerate(ITEMS) if i.kind == "pod"]]))
                    gx, gy = ((-0.04, -0.04), (0.04, -0.04), (-0.04, 0.04), (0.04, 0.04))[n % 4]
                    pos = (c[0] + gx, c[1] + gy, c[2] + 0.02 + 0.03 * (n // 4))
                elif tops and rng.uniform() < 0.2:
                    (cx, cy), (hx, hy), z = tops[rng.integers(len(tops))]
                    pos = (cx + rng.uniform(-hx + 0.1, hx - 0.1), cy + rng.uniform(-hy + 0.1, hy - 0.1), z + 0.05)
                else:
                    for _ in range(80):
                        p = np.array([rng.uniform(-4.4, 4.4), rng.uniform(-2.9, 2.9)])
                        if H.room_of(p) in ("bedroom",) or rng.uniform() < 0.25:
                            if is_free(p, 0.15) and all(np.linalg.norm(p - q) > 0.12 for q in placed):
                                pos = (p[0], p[1], height_at(*p) + 0.06)
                                placed.append(p)
                                break
            if pos is None:
                qpos[qa:qa + 7] = [PARK[0] - 1.0 + 0.3 * k, PARK[1], PARK[2], 1, 0, 0, 0]
                continue
            qpos[qa:qa + 7] = [*pos, *quat]
            self.active[k] = True
            want[it.kind] -= 1

    def locate(self, data) -> list:
        """For each item: the container it is in (name), or None."""
        out = []
        boxes = {}
        for name, s in self.sites.items():
            boxes[name] = (data.site_xpos[s], data.site_xmat[s].reshape(3, 3), self.m.site_size[s])
        for k, b in enumerate(self.body):
            where = None
            if self.active[k]:
                p = data.xpos[b]
                for name, (c, R, hs) in boxes.items():
                    loc = R.T @ (p - c)
                    if np.all(np.abs(loc) <= hs):
                        where = name
                        break
            out.append(where)
        return out

    def tidy_count(self, data) -> tuple[int, dict]:
        where = self.locate(data)
        done = 0
        detail = {}
        for k, it in enumerate(ITEMS):
            if not self.active[k]:
                continue
            ok = where[k] in GOALS[it.kind]
            done += ok
            detail[it.name] = where[k]
        return done, detail
