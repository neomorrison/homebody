"""The floor of a real home, regenerated every episode.

``terrain_level`` scales everything from a perfect lab floor (0) to an ordinary
lived-in home (1) and beyond (2+ for robustness training):

* **Uneven floor.** A heightfield with a slight tilt, slow undulation and a few
  warped spots. Real floors are flat to a few millimetres over a few metres;
  level 1 is about 3 mm RMS of bumps (up to 6 mm) on a floor tilted up to 0.5 degree.
* **Surfaces.** The floor is tile, laminate, hardwood, concrete or carpet, and
  patches lie on it: rugs, carpet offcuts, rubber mats, wet spots and dusty
  spots. Each has its own sole-to-surface friction and softness.
* **Obstacles you can step on.** Door sills (8 and 16 mm), cables, and loose
  clutter (a toy block, a ball, a book) that slides or rolls when kicked.

Friction values are coefficients between a rubber sole and the surface, the
number the contact actually uses: every world surface has a higher contact
priority than the robot, so the surface decides (``robot_xml`` keeps robot
parts at priority 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

# sole-on-surface friction ranges, softness (MuJoCo solref time constant, s), colour
SURFACES = {
    "tile":       dict(mu=(0.35, 0.55), soft=0.02, rgba=(0.86, 0.86, 0.83, 1)),
    "laminate":   dict(mu=(0.45, 0.65), soft=0.02, rgba=(0.78, 0.66, 0.50, 1)),
    "hardwood":   dict(mu=(0.50, 0.75), soft=0.02, rgba=(0.62, 0.45, 0.30, 1)),
    "concrete":   dict(mu=(0.60, 0.85), soft=0.02, rgba=(0.62, 0.62, 0.60, 1)),
    "carpet":     dict(mu=(0.80, 1.10), soft=0.05, rgba=(0.45, 0.50, 0.58, 1)),
    "rug":        dict(mu=(0.70, 1.00), soft=0.035, rgba=(0.60, 0.22, 0.22, 1)),
    "rubber_mat": dict(mu=(1.00, 1.30), soft=0.03, rgba=(0.15, 0.15, 0.15, 1)),
    "wet":        dict(mu=(0.12, 0.30), soft=0.02, rgba=(0.70, 0.80, 0.90, 0.6)),
    "dusty":      dict(mu=(0.25, 0.40), soft=0.02, rgba=(0.90, 0.88, 0.80, 0.5)),
}
FLOOR_TYPES = ["tile", "laminate", "hardwood", "concrete", "carpet"]
PATCH_TYPES = ["rug", "rug", "carpet", "rubber_mat", "wet", "dusty"]
NOMINAL = dict(mu=0.6, soft=0.02)

# patch half-sizes (x, y, thickness/2); fixed at compile, placed per episode
PATCH_SIZES = [(0.9, 0.6, 0.003), (1.1, 0.7, 0.004), (0.45, 0.3, 0.003), (0.3, 0.22, 0.0005), (0.5, 0.35, 0.0005)]
SILL_HEIGHTS = [0.004, 0.008]          # half-heights: 8 mm and 16 mm sills
N_CABLES = 2
CLUTTER = [  # name, geom xml, mass
    ("block", 'type="box" size="0.02 0.02 0.02"', 0.03),
    ("ball", 'type="sphere" size="0.03"', 0.05),
    ("book", 'type="box" size="0.1 0.07 0.012"', 0.3),
]

HF_RES = 0.04          # m per heightfield cell
HF_BASE = 0.05         # floor at z=0 sits this far above the heightfield's zero
HF_TOP = 0.10          # max representable height above the base


@dataclass
class TerrainState:
    level: float
    floor: str
    floor_mu: float
    patches: list        # (type, mu)
    tilt_deg: float
    bump_rms_mm: float


def _f(*v):
    return " ".join(f"{x:.5g}" for x in v)


def hfield_shape(room_size, margin=0.15):
    rx, ry = room_size[0] / 2 + margin, room_size[1] / 2 + margin
    return rx, ry, int(round(2 * ry / HF_RES)) + 1, int(round(2 * rx / HF_RES)) + 1


def assets_xml(room_size) -> str:
    rx, ry, nrow, ncol = hfield_shape(room_size)
    return f'<hfield name="floor" nrow="{nrow}" ncol="{ncol}" size="{_f(rx, ry, HF_TOP, 0.05)}"/>'


def world_xml() -> str:
    """Static floor pieces. Goes inside <worldbody>."""
    surf = 'priority="2" condim="4" friction="0.6 0.02 0.001"'
    out = [f'<geom name="floor" type="hfield" hfield="floor" pos="0 0 {-HF_BASE}" material="floor" {surf}/>',
           # out-of-sight shelf where unused clutter waits
           f'<geom name="storage" type="box" size="1 1 0.05" pos="30 0 -1" {surf} rgba="0 0 0 0"/>']
    for k, (x, y, z) in enumerate(PATCH_SIZES):
        out.append(f'<body name="patch_{k}" pos="30 0 -2"><geom name="patch_{k}" type="box" size="{_f(x, y, z)}" '
                   f'pos="0 0 {z:.4f}" {surf} rgba="0.6 0.2 0.2 1"/></body>')
    for k, h in enumerate(SILL_HEIGHTS):
        out.append(f'<body name="sill_{k}" pos="30 0 -2"><geom name="sill_{k}" type="box" size="{_f(0.45, 0.025, h)}" '
                   f'pos="0 0 {h:.4f}" {surf} rgba="0.55 0.45 0.35 1"/></body>')
    for k in range(N_CABLES):
        out.append(f'<body name="cable_{k}" pos="30 0 -2"><geom name="cable_{k}" type="capsule" size="0.004" '
                   f'fromto="-0.4 0 0.004 0.4 0 0.004" {surf} rgba="0.1 0.1 0.1 1"/></body>')
    return "\n    ".join(out)


def clutter_xml() -> str:
    """Loose objects with free joints. Goes after the robot so the robot's root stays qpos[0:7]."""
    surf = 'priority="2" condim="3" friction="0.5 0.01 0.001"'
    return "\n    ".join(
        f'<body name="clutter_{n}" pos="{30 + 0.3 * k} 0 -0.8"><freejoint name="clutter_{n}"/>'
        f'<geom name="clutter_{n}" {g} mass="{m}" {surf} rgba="0.2 0.5 0.8 1"/></body>'
        for k, (n, g, m) in enumerate(CLUTTER)
    )


def _smooth_noise(rng, nrow, ncol, sigma_cells):
    """Gaussian-filtered white noise with unit std (FFT, periodic; fine for a floor)."""
    w = rng.normal(size=(nrow, ncol))
    fy = np.fft.fftfreq(nrow)[:, None]
    fx = np.fft.fftfreq(ncol)[None, :]
    k = np.exp(-2 * (math.pi * sigma_cells) ** 2 * (fx * fx + fy * fy))
    out = np.real(np.fft.ifft2(np.fft.fft2(w) * k))
    return out / (out.std() + 1e-12)


class Terrain:
    def __init__(self, model, room_size):
        m = model
        self.m = m
        self.rx, self.ry, self.nrow, self.ncol = hfield_shape(room_size)
        self.room = room_size
        name = lambda t, n: mujoco.mj_name2id(m, t, n)
        B, G, J = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_JOINT
        self.hf = name(mujoco.mjtObj.mjOBJ_HFIELD, "floor")
        self.floor_geom = name(G, "floor")
        self.floor_mat = name(mujoco.mjtObj.mjOBJ_MATERIAL, "floor")
        self.patch = [(name(B, f"patch_{k}"), name(G, f"patch_{k}")) for k in range(len(PATCH_SIZES))]
        self.sill = [name(B, f"sill_{k}") for k in range(len(SILL_HEIGHTS))]
        self.cable = [name(B, f"cable_{k}") for k in range(N_CABLES)]
        self.clutter = [(m.jnt_qposadr[name(J, f"clutter_{n}")], m.jnt_dofadr[name(J, f"clutter_{n}")])
                        for n, _, _ in CLUTTER]
        self.heights = np.zeros((self.nrow, self.ncol))
        self.version = 0
        xs = np.linspace(-self.rx, self.rx, self.ncol)
        ys = np.linspace(-self.ry, self.ry, self.nrow)
        self.X, self.Y = np.meshgrid(xs, ys)

    # --------------------------------------------------------------- heights
    def height_at(self, x, y) -> float:
        """Bilinear floor height (m) at world x, y."""
        c = (x + self.rx) / (2 * self.rx) * (self.ncol - 1)
        r = (y + self.ry) / (2 * self.ry) * (self.nrow - 1)
        c = min(max(c, 0), self.ncol - 1.001)
        r = min(max(r, 0), self.nrow - 1.001)
        c0, r0 = int(c), int(r)
        fc, fr = c - c0, r - r0
        h = self.heights
        return float((1 - fr) * ((1 - fc) * h[r0, c0] + fc * h[r0, c0 + 1])
                     + fr * ((1 - fc) * h[r0 + 1, c0] + fc * h[r0 + 1, c0 + 1]))

    def generate_heights(self, rng, level):
        h = np.zeros((self.nrow, self.ncol))
        tilt = 0.0
        if level > 0:
            tilt = level * rng.uniform(0, 0.5)                       # degrees
            a = rng.uniform(0, 2 * math.pi)
            g = math.tan(math.radians(tilt))
            h += g * (math.cos(a) * self.X + math.sin(a) * self.Y)
            h += level * 0.003 * _smooth_noise(rng, self.nrow, self.ncol, rng.uniform(6, 15))
            for _ in range(rng.integers(0, int(6 * min(level, 2)) + 1)):  # warped boards, settled slab
                cx, cy = rng.uniform(-self.rx, self.rx), rng.uniform(-self.ry, self.ry)
                s = rng.uniform(0.05, 0.15)
                h += level * rng.uniform(-0.006, 0.006) * np.exp(-((self.X - cx) ** 2 + (self.Y - cy) ** 2) / (2 * s * s))
        h -= h.mean()
        self.heights = np.clip(h, -HF_BASE + 0.001, HF_TOP - HF_BASE - 0.001)
        self.tilt = tilt
        # bumpiness without the tilt: residual of a least-squares plane
        A = np.stack([self.X.ravel(), self.Y.ravel(), np.ones(self.X.size)], 1)
        coef, *_ = np.linalg.lstsq(A, self.heights.ravel(), rcond=None)
        self.bump_rms = float((self.heights.ravel() - A @ coef).std())

    def flatten(self, x, y, yaw, half_x, half_y, pad=0.06):
        """Level a rectangle (furniture, dock) to its mean height and return that height."""
        c, s = math.cos(yaw), math.sin(yaw)
        lx = c * (self.X - x) + s * (self.Y - y)
        ly = -s * (self.X - x) + c * (self.Y - y)
        inside = (np.abs(lx) < half_x + pad) & (np.abs(ly) < half_y + pad)
        if not inside.any():
            return self.height_at(x, y)
        z = float(self.heights[inside].mean())
        self.heights[inside] = z
        return z

    def upload(self):
        """Write heights into the model (call after all flatten() calls)."""
        data = (self.heights + HF_BASE) / HF_TOP
        self.m.hfield_data[:] = np.clip(data, 0, 1).ravel()
        self.version += 1

    # --------------------------------------------------------------- surfaces
    def _surface(self, rng, kind, level):
        s = SURFACES[kind]
        mu = rng.uniform(*s["mu"])
        mu = NOMINAL["mu"] + min(level, 1.0) * (mu - NOMINAL["mu"])
        soft = NOMINAL["soft"] + min(level, 1.0) * (s["soft"] - NOMINAL["soft"])
        return mu, soft, s["rgba"]

    def _set_geom_surface(self, g, mu, soft, rgba=None):
        m = self.m
        m.geom_friction[g, 0] = mu
        m.geom_solref[g] = [soft, 1.0]
        if rgba is not None:
            m.geom_rgba[g] = rgba

    def randomize(self, rng, level, taken, set_body, sole_scale=1.0):
        """Surfaces, patches, sills, cables and clutter. ``taken`` = [(xy, radius)] to keep clear
        (the dock, furniture). Returns a TerrainState summary. Call after generate_heights() and
        before upload(): patches level the floor under themselves."""
        m = self.m
        X, Y = self.room[0] / 2, self.room[1] / 2
        floor = "tile" if level <= 0 else str(rng.choice(FLOOR_TYPES))
        mu, soft, rgba = self._surface(rng, floor, level)
        self._set_geom_surface(self.floor_geom, mu * sole_scale, soft)
        m.mat_rgba[self.floor_mat] = rgba
        state = TerrainState(level, floor, mu * sole_scale, [], self.tilt, 1000 * self.bump_rms)

        def free_spot(r, clear=0.15):
            for _ in range(40):
                p = np.array([rng.uniform(-X + r, X - r), rng.uniform(-Y + r, Y - r)])
                if all(np.linalg.norm(p - q) > rq + clear for q, rq in taken):
                    return p
            return None

        def hide(b):
            m.body_pos[b] = [30, 0, -2]

        for b, g in self.patch:
            hx, hy, hz = m.geom_size[g]
            p = free_spot(0.5 * max(hx, hy)) if rng.uniform() < level else None
            if p is None:
                hide(b)
                continue
            kind = str(rng.choice(PATCH_TYPES))
            pmu, psoft, prgba = self._surface(rng, kind, level)
            self._set_geom_surface(g, pmu * sole_scale, psoft, prgba)
            yaw = rng.uniform(0, math.pi)
            set_body(b, p, yaw)
            m.body_pos[b][2] = self.flatten(p[0], p[1], yaw, hx, hy, pad=0.0)   # a rug lies flat
            state.patches.append((kind, round(pmu * sole_scale, 2)))
        for b in self.sill + self.cable:
            p = free_spot(0.45) if rng.uniform() < 0.6 * level else None
            if p is None:
                hide(b)
                continue
            set_body(b, p, rng.uniform(0, math.pi))
            m.body_pos[b][2] = self.height_at(*p) - 0.001
        return state

    def place_clutter(self, rng, level, taken, qpos, qvel):
        X, Y = self.room[0] / 2, self.room[1] / 2
        for k, (qa, va) in enumerate(self.clutter):
            qvel[va:va + 6] = 0
            p = None
            if rng.uniform() < level:
                for _ in range(40):
                    c = np.array([rng.uniform(-X + 0.2, X - 0.2), rng.uniform(-Y + 0.2, Y - 0.2)])
                    if all(np.linalg.norm(c - q) > rq + 0.1 for q, rq in taken):
                        p = c
                        break
            if p is None:
                qpos[qa:qa + 7] = [30 - 0.6 + 0.3 * k, 0, -0.85, 1, 0, 0, 0]
                continue
            yaw = rng.uniform(-math.pi, math.pi)
            qpos[qa:qa + 7] = [p[0], p[1], self.height_at(*p) + 0.04, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
