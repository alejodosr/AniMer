"""Build a ground calibration from ZoeDepth alone — no clicking, no tiles.

The clicked calibration pins the plane's SHAPE from four image points (which
are reliable) but its METRES came from counting tiles and assuming a tile
size (which is rough). This replaces the whole thing with a plane fitted to a
metric depth map, so the absolute scale comes from the depth model instead of
the tile guess.

Emits the same JSON contract calibrate_ground.py writes, so every downstream
stage consumes it unchanged.

Construction: a plane (unit normal n, offset d) in camera coordinates defines
a world frame with Z along n and the origin at the foot of the perpendicular
from the camera. The in-plane axes are arbitrary — a rotation about the
vertical changes nothing measurable (distances, heights, skate are all
invariant) — so X is taken as the camera's forward direction projected onto
the plane. Then

    H_inv = K [x_w | y_w | o_c]        (ground metres -> image pixels)

which is Zhang's factorisation run backwards, and H is its inverse.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from depth_ground import sample_frames, fit_plane_ransac


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--ref-calib", required=True,
                   help="only for img_size and focal_px; the plane is NOT taken "
                        "from it")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--floor-frac", type=float, default=0.45)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--cache", default="/media/SHARED_DATA/postcapitalistrobots/"
                   "animer/paw/depthcache")
    args = p.parse_args()

    import os
    os.environ["TORCH_HOME"] = args.cache
    import torch

    ref = json.loads(Path(args.ref_calib).read_text())
    W, H = ref["img_size"]
    focal = float(ref["focal_px"])
    frames, idx = sample_frames(args.video, args.frames, W, H)
    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True,
                           trust_repo=True).cuda().eval()

    ys, xs = np.mgrid[0:H:args.stride, 0:W:args.stride]
    keep = ys.ravel() > H * (1 - args.floor_frac)
    u = xs.ravel()[keep].astype(np.float64)
    v = ys.ravel()[keep].astype(np.float64)

    normals, dists = [], []
    for f in frames:
        with torch.no_grad():
            t = torch.from_numpy(f).permute(2, 0, 1)[None].float().cuda() / 255.0
            dm = model.infer(t)[0, 0].cpu().numpy()
        z = dm[ys.ravel()[keep], xs.ravel()[keep]]
        P = np.stack([(u - W / 2.0) * z / focal,
                      (v - H / 2.0) * z / focal, z], 1)
        n, d, inl = fit_plane_ransac(P)
        if n is None:
            continue
        if n[1] > 0:            # want the normal pointing UP in camera coords
            n, d = -n, -d       # camera y is DOWN, so up is -y
        normals.append(n)
        dists.append(abs(d))
    if not normals:
        raise SystemExit("no plane fitted")
    n = np.mean(normals, axis=0)
    n /= np.linalg.norm(n)
    h = float(np.median(dists))
    print(f"depth plane: normal {n.round(3)}, camera height {h:.3f} m "
          f"(over {len(dists)} frames, spread {min(dists):.3f}-{max(dists):.3f})")

    # world frame: Z = n (up), X = camera forward projected into the plane
    fwd = np.array([0.0, 0.0, 1.0])
    x_w = fwd - (fwd @ n) * n
    x_w /= np.linalg.norm(x_w)
    y_w = np.cross(n, x_w)
    o_c = h * (-n)                     # foot of the perpendicular, camera coords
    K = np.array([[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1.0]])
    H_inv = K @ np.stack([x_w, y_w, o_c], axis=1)
    H_inv = H_inv / H_inv[2, 2]
    Hm = np.linalg.inv(H_inv)

    # sanity: a synthetic ground rectangle must round-trip
    test = np.array([[0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
    px = (H_inv @ test.T).T
    px = px[:, :2] / px[:, 2:3]
    back = (Hm @ np.concatenate([px, np.ones((3, 1))], 1).T).T
    back = back[:, :2] / back[:, 2:3]
    err = np.abs(back - test[:, :2]).max()
    print(f"round-trip error {err:.2e} m (must be ~0)")

    poly = np.array([[W * 0.05, H * 0.55], [W * 0.95, H * 0.55],
                     [W * 0.95, H * 0.98], [W * 0.05, H * 0.98]])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "camera": Path(args.out).stem, "source_video": args.video, "frame": 0,
        "img_size": [W, H], "max_side": ref.get("max_side", 1280),
        "pixels": [], "world_m": [],
        "H": Hm.tolist(), "H_inv": H_inv.tolist(),
        "residual_px": [0.0], "residual_max_px": 0.0,
        "validity_polygon": poly.tolist(), "validity_dilate": 1.0,
        "focal_px": focal,
        "depth_plane_normal_cam": n.tolist(),
        "depth_camera_height_m": h,
        "note": "ground plane from ZoeDepth metric depth; absolute scale is "
                "the depth model's, NOT a tile-count assumption",
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
