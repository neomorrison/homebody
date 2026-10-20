"""Turn a phone LiDAR scan (OBJ from 3d Scanner App or similar) into the sim's home map.

    python scripts/import_scan.py scans/2026-10-19/textured_output.obj

Writes homebody/assets/home_scan.npz:
    H       raw height map (m above the floor, 4 cm cells, ceiling ignored), -1 = never seen
    x0, y0  map origin in the aligned scan frame (m)
    rot     degrees the scan was turned so the walls run along x and y
    floor   scan height of the floor (subtracted)
and homebody/assets/home_scan.png, a preview. homebody/home.py measures the rooms from
this map and rebuilds them as simple solids; the floor and rug textures in
homebody/assets/textures/ were cut from the scan's ortho renders.

Also writes scans/<date>/scan_tex.png (the texture as PNG), for cutting new textures.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_obj(path):
    V, F = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                V.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(V), np.array(F)


def best_rotation(P):
    """Yaw (deg) that makes the wall points pile up on as few x and y lines as possible."""
    wall = P[(P[:, 2] > 0.6) & (P[:, 2] < 1.8)]
    best = (-1, 0.0)
    for deg in np.arange(0, 90, 0.25):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        x = c * wall[:, 0] - s * wall[:, 1]
        y = s * wall[:, 0] + c * wall[:, 1]
        hx, _ = np.histogram(x, bins=np.arange(x.min(), x.max(), 0.03))
        hy, _ = np.histogram(y, bins=np.arange(y.min(), y.max(), 0.03))
        score = float((hx.astype(float) ** 2).sum() + (hy.astype(float) ** 2).sum())
        if score > best[0]:
            best = (score, deg)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj")
    ap.add_argument("--res", type=float, default=0.04)
    ap.add_argument("--ceiling-cut", type=float, default=2.1, help="ignore geometry above this (m)")
    a = ap.parse_args()

    V, F = load_obj(a.obj)
    P = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)            # y-up scan -> z-up
    rot = best_rotation(P)
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    P = np.stack([c * P[:, 0] - s * P[:, 1], s * P[:, 0] + c * P[:, 1], P[:, 2]], 1)
    T = P[F]
    pts = [P] + [np.einsum("k,fkd->fd", np.array(w), T)        # face samples fill big flat faces
                 for w in ([1 / 3] * 3, [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3])]
    pts = np.concatenate(pts)
    floor = float(np.percentile(pts[:, 2], 2))
    pts[:, 2] -= floor
    x0, y0 = pts[:, 0].min() - 0.2, pts[:, 1].min() - 0.2
    nx = int((pts[:, 0].max() + 0.2 - x0) / a.res) + 1
    ny = int((pts[:, 1].max() + 0.2 - y0) / a.res) + 1
    ix = ((pts[:, 0] - x0) / a.res).astype(int)
    iy = ((pts[:, 1] - y0) / a.res).astype(int)
    H = np.full((ny, nx), -1.0, np.float32)
    low = pts[:, 2] < a.ceiling_cut
    np.maximum.at(H, (iy[low], ix[low]), pts[low, 2].astype(np.float32))

    out = os.path.join(ROOT, "homebody", "assets", "home_scan.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, H=H, x0=x0, y0=y0, res=a.res, rot=rot, floor=floor,
                        source=os.path.relpath(a.obj, ROOT).replace("\\", "/"))
    img = np.clip(H / a.ceiling_cut, 0, 1)
    rgb = np.stack([img] * 3, -1)
    rgb[H < 0] = [0.3, 0, 0]
    Image.fromarray((rgb[::-1] * 255).astype(np.uint8)).resize((nx * 3, ny * 3), Image.NEAREST).save(
        out.replace(".npz", ".png"))
    tex = os.path.join(os.path.dirname(a.obj), "textured_output.jpg")
    if os.path.exists(tex):
        Image.open(tex).convert("RGB").resize((4096, 4096), Image.LANCZOS).save(
            os.path.join(os.path.dirname(a.obj), "scan_tex.png"))
    print(f"rotation {rot:.2f} deg, floor {floor:.3f} m, grid {H.shape}, x0 {x0:.2f}, y0 {y0:.2f} -> {out}")


if __name__ == "__main__":
    sys.exit(main())
