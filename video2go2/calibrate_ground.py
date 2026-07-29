"""Phase A — ground-plane calibration from four clicked points.

Four image<->plane correspondences fully determine a homography (8 DOF, two
equations per point). The homography absorbs the intrinsics, so there is no
focal length to measure and no lens calibration to run. Any pixel known to lie
on the ground maps to a ground position directly, with no dependence on
apparent size -- which is exactly what monocular depth cannot give us.

The homography says nothing about points off the plane. That is fine: the only
points we need in world coordinates are feet in contact.

One JSON per camera setup, reused across clips shot from that setup.

SHOOTING AND CLICKING GUIDANCE
------------------------------
* Spread the four points toward the corners of the region the animal
  traverses. Bunched points make the homography numerically fragile
  everywhere else.
* The points must be genuinely coplanar and genuinely on the ground. Kerbs,
  doorsteps, wall bases with skirting, and paving lips are the classic
  failure -- a skirting board's top edge is centimetres above the floor and
  will tilt the whole plane.
* Camera at waist height or above, angled down. A camera at animal height
  looking horizontally makes the plane near edge-on and the solution
  degenerate. This is the single biggest quality lever and it is free.
* No wide-angle. Barrel distortion breaks the straight-lines-stay-straight
  assumption, worst at the frame edges. Phone at 1x.
* Have the animal traverse ACROSS the frame, not toward the camera. Motion
  along the viewing direction is the worst-conditioned case: on dog_1.mov,
  10 cm of travel in depth moves a paw about 1 pixel, against 37 pixels for
  the same travel sideways.
* Distances only need to be roughly right. Absolute scale is re-fitted later
  from the animal's own proportions; these fix the plane's SHAPE, so what
  matters is the ratio between them, not their absolute values. A tiled floor
  is ideal -- count tiles rather than pacing the room.
"""
from pathlib import Path
import argparse
import json

import numpy as np


def frame_size(src_w, src_h, max_side):
    """Match animer_infer.py exactly, so pixels mean the same thing."""
    s = 1.0
    if max_side > 0 and max(src_w, src_h) > max_side:
        s = max_side / max(src_w, src_h)
    return int(round(src_w * s)) // 2 * 2, int(round(src_h * s)) // 2 * 2


def grab_frame(video, index, max_side):
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {index}")
    W, H = frame_size(sw, sh, max_side)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), W, H


def check_configuration(pixels, world, min_shape=0.05):
    """Reject near-degenerate point sets before fitting.

    If three of the four points are near-collinear the homography is
    underdetermined, but DLT still returns a solution that fits all four
    points EXACTLY -- residual 0.00 px -- while being wrong everywhere else.
    The residual check is blind to this, so it has to be caught here.

    Measure: for every triple, |cross| / diameter^2. That is ~0.87 for an
    equilateral triangle and 0 for collinear points.
    """
    problems = []
    for name, pts in (("image", pixels), ("world", world)):
        p = np.asarray(pts, float)
        n = len(p)
        diam = max(np.linalg.norm(p[i] - p[j])
                   for i in range(n) for j in range(i + 1, n))
        if diam <= 0:
            problems.append(f"{name} points are coincident")
            continue
        def cross2(a, b):
            return float(a[0] * b[1] - a[1] * b[0])
        worst = min(
            abs(cross2(p[j] - p[i], p[k] - p[i])) / diam ** 2
            for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n))
        if worst < min_shape:
            problems.append(
                f"three {name} points are nearly collinear (shape {worst:.3f} "
                f"< {min_shape}); the fit will look perfect and be wrong")
    return problems


def fit(pixels, world, strict=True):
    """Homography image->ground, its inverse, and the reprojection residual."""
    import cv2
    problems = check_configuration(pixels, world)
    if problems and strict:
        raise SystemExit("degenerate point configuration:\n  - "
                         + "\n  - ".join(problems)
                         + "\nSpread the points toward the corners of the "
                           "region the animal traverses.")
    pix = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
    wld = np.asarray(world, np.float64).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pix, wld, method=0)
    if H is None:
        raise SystemExit("findHomography failed -- are the points degenerate "
                         "(three collinear, or all bunched)?")
    Hinv = np.linalg.inv(H)
    back = cv2.perspectiveTransform(wld, Hinv).reshape(-1, 2)
    resid = np.linalg.norm(back - np.asarray(pixels, np.float64), axis=1)
    return H, Hinv, resid


def solve_focal(H_inv, cx, cy):
    """Focal length in pixels, from the homography of a KNOWN rectangle.

    Zhang: the ground->image homography factors as lambda*K[r1 r2 t], and r1,r2
    are columns of a rotation, so they are equal in length and orthogonal.

    Only the equal-length condition is used to SOLVE. The orthogonality form
    also gives a closed form for f^2, but it divides by h1_z*h2_z, and h1_z is
    ~0 whenever the world X axis runs parallel to the image rows -- which is
    the normal case when you click a near edge that looks horizontal. On the
    two existing calibrations that denominator is ~1e-18 and the estimate
    explodes. Orthogonality is used as a RESIDUAL instead: at the solved f, how
    far from perpendicular the two recovered axes are. That is a free check on
    the clicked points and the assumed tile counts.

    Returns (focal_px, orthogonality_cosine). Cosine near 0 is good.
    """
    Hi = np.asarray(H_inv, float)
    a, b, c = Hi[:, 0]
    d, e, g = Hi[:, 1]
    ax, ay = a - cx * c, b - cy * c
    dx, dy = d - cx * g, e - cy * g
    den = g * g - c * c
    scale = max(abs(a), abs(b), abs(d), abs(e), 1e-12)
    if abs(den) < 1e-9 * (scale / max(abs(a), 1e-12)) ** 0 * 1e-9:
        return None, None
    v = ((ax ** 2 + ay ** 2) - (dx ** 2 + dy ** 2)) / den
    if not np.isfinite(v) or v <= 0:
        return None, None
    f = float(np.sqrt(v))
    r1 = np.array([ax / f, ay / f, c])
    r2 = np.array([dx / f, dy / f, g])
    cos = float(abs(np.dot(r1, r2)) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
    return f, cos


def validity_polygon(pixels, dilate=1.25):
    """Convex hull of the clicked points, scaled about its centroid.

    Ground mapping outside this is extrapolation. Downstream flags rather than
    rejects, because a foot slightly outside is still informative.
    """
    import cv2
    pts = np.asarray(pixels, np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    c = hull.mean(axis=0)
    return (c + (hull - c) * dilate)


def to_ground(H, uv):
    """(...,2) pixels -> (...,2) ground metres."""
    uv = np.asarray(uv, np.float64)
    flat = uv.reshape(-1, 2)
    h = np.concatenate([flat, np.ones((len(flat), 1))], axis=1) @ H.T
    return (h[:, :2] / h[:, 2:3]).reshape(uv.shape)


def click_points(img, n=4):
    import cv2
    pts = []
    disp = img.copy()
    win = "click 4 ground points (spread wide) -- u undo, enter accept, esc quit"

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((float(x), float(y)))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, img.shape[1]), min(1000, img.shape[0]))
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = img.copy()
        for i, (x, y) in enumerate(pts):
            cv2.drawMarker(disp, (int(x), int(y)), (0, 235, 0),
                           cv2.MARKER_CROSS, 22, 2)
            cv2.putText(disp, str(i + 1), (int(x) + 9, int(y) - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 235, 0), 2)
        if len(pts) > 1:
            cv2.polylines(disp, [np.int32(pts)], len(pts) == n, (0, 180, 255), 1)
        cv2.imshow(win, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k in (13, 10) and len(pts) == n:
            break
        elif k == 27:
            cv2.destroyAllWindows()
            raise SystemExit("cancelled")
    cv2.destroyAllWindows()
    return pts


def ask_world(pts):
    print("\nNow the rough real-world geometry of those 4 points.")
    print("Give them as X,Y metres on the floor, in the order you clicked.")
    print("Put point 1 at the origin and let point 2 define the +X direction;")
    print("that is a convention, not a measurement. Only the ratios matter.\n")
    print("For a rectangle of width W (across) and length L (along), clicking")
    print("near-left, near-right, far-right, far-left, you would enter:")
    print("    0,0    W,0    W,L    0,L\n")
    world = []
    for i in range(len(pts)):
        while True:
            s = input(f"  point {i+1} at pixel ({pts[i][0]:.0f},{pts[i][1]:.0f}) "
                      f"-> X,Y metres: ").strip().replace(" ", "")
            try:
                x, y = (float(v) for v in s.split(","))
                world.append((x, y))
                break
            except Exception:
                print("    need two numbers like  1.2,0")
    return world


def draw_overlay(img, H, Hinv, pixels, poly, resid, step=0.5, out=None):
    """Project a metric grid onto the floor. THE verification: the grid must
    lie along the real tile lines and stay square as it recedes."""
    import cv2
    vis = img.copy()
    g = to_ground(H, np.asarray(pixels, np.float64))
    lo = g.min(axis=0) - 2.0
    hi = g.max(axis=0) + 2.0

    def seg(p, q):
        pq = cv2.perspectiveTransform(
            np.array([[p], [q]], np.float64).reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        return tuple(np.int32(pq[0])), tuple(np.int32(pq[1]))

    x = np.floor(lo[0] / step) * step
    while x <= hi[0]:
        try:
            a, b = seg((x, lo[1]), (x, hi[1]))
            cv2.line(vis, a, b, (0, 200, 255), 1, cv2.LINE_AA)
        except Exception:
            pass
        x += step
    y = np.floor(lo[1] / step) * step
    while y <= hi[1]:
        try:
            a, b = seg((lo[0], y), (hi[0], y))
            cv2.line(vis, a, b, (255, 190, 0), 1, cv2.LINE_AA)
        except Exception:
            pass
        y += step

    cv2.polylines(vis, [np.int32(poly)], True, (0, 0, 255), 2)
    for i, (px, py) in enumerate(pixels):
        cv2.drawMarker(vis, (int(px), int(py)), (0, 235, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(vis, f"{i+1} ({resid[i]:.1f}px)", (int(px) + 9, int(py) - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 235, 0), 2)
    cv2.putText(vis, f"grid {step} m   max residual {resid.max():.2f} px",
                (12, vis.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), vis)
        print(f"wrote {out}")
    return vis


def selftest():
    """Recover a known homography from a synthetic camera.

    Correctness here does not depend on anyone identifying real tile corners,
    which is why this exists.
    """
    rng = np.random.default_rng(0)
    K = np.array([[900.0, 0, 640.0], [0, 900.0, 400.0], [0, 0, 1.0]])
    # camera 2.2 m up, pitched 35 deg down, looking along +Y
    t = np.deg2rad(35.0)
    Rz = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    flip = np.array([[1, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])   # world Z-up -> cam Y-down
    R = Rz @ flip
    C = np.array([0.0, -3.0, 2.2])
    world = np.array([[-1.0, 1.0], [1.0, 1.0], [1.4, 4.0], [-1.4, 4.0]])
    P3 = np.stack([world[:, 0], world[:, 1], np.zeros(len(world))], axis=1)
    cam = (P3 - C) @ R.T
    uv = (cam @ K.T)[:, :2] / (cam @ K.T)[:, 2:3]

    H, Hinv, resid = fit(uv, world)
    back = to_ground(H, uv)
    err = np.linalg.norm(back - world, axis=1)
    print(f"selftest: reprojection residual max {resid.max():.3e} px, "
          f"ground error max {err.max():.3e} m")

    # a point NOT among the four must also map correctly
    probe_w = np.array([[0.3, 2.4], [-0.8, 3.1]])
    P3 = np.stack([probe_w[:, 0], probe_w[:, 1], np.zeros(len(probe_w))], axis=1)
    cam = (P3 - C) @ R.T
    puv = (cam @ K.T)[:, :2] / (cam @ K.T)[:, 2:3]
    perr = np.linalg.norm(to_ground(H, puv) - probe_w, axis=1)
    print(f"selftest: held-out ground error max {perr.max():.3e} m")
    # Tolerances are "numerically exact for our purposes", not machine epsilon:
    # findHomography normalises and solves by DLT, so residuals land around
    # 1e-5 px. Anything at that level means the geometry is right; a real
    # mistake shows up at 1e-1 px or worse, orders away from these bounds.
    ok = resid.max() < 1e-3 and err.max() < 1e-6 and perr.max() < 1e-6
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--max-side", type=int, default=1280,
                   help="must match animer_infer.py, or pixels disagree")
    p.add_argument("--camera", default=None, help="name for the calibration")
    p.add_argument("--out", default=None, help="calib json path")
    p.add_argument("--points", default=None,
                   help="non-interactive: 'u,v;u,v;u,v;u,v'")
    p.add_argument("--world", default=None,
                   help="non-interactive: 'x,y;x,y;x,y;x,y' in metres")
    p.add_argument("--grid", type=float, default=0.5, help="overlay grid step (m)")
    p.add_argument("--overlay", default=None, help="write the verification png")
    p.add_argument("--verify", default=None, help="load a calib json and re-draw")
    p.add_argument("--dilate", type=float, default=1.25)
    p.add_argument("--focal-px", type=float, default=None,
                   help="calibrated focal length in pixels for THIS frame "
                        "size. Recorded so downstream never accidentally uses "
                        "AniMer's assumed 5000 px, which is a 14.6 deg FOV.")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    if args.verify:
        d = json.loads(Path(args.verify).read_text())
        img, W, H_ = grab_frame(args.video, args.frame, args.max_side)
        if [W, H_] != d["img_size"]:
            raise SystemExit(f"frame is {W}x{H_} but calib is for "
                             f"{d['img_size']} -- --max-side must match")
        Hm = np.array(d["H"])
        draw_overlay(img, Hm, np.array(d["H_inv"]), d["pixels"],
                     np.array(d["validity_polygon"]),
                     np.array(d["residual_px"]), args.grid,
                     args.overlay or "verify.png")
        return

    if not args.video:
        raise SystemExit("need --video (or --selftest)")
    img, W, H_ = grab_frame(args.video, args.frame, args.max_side)

    if args.points:
        pixels = [tuple(float(v) for v in s.split(","))
                  for s in args.points.split(";")]
        if not args.world:
            raise SystemExit("--points needs --world")
        world = [tuple(float(v) for v in s.split(","))
                 for s in args.world.split(";")]
    else:
        pixels = click_points(img)
        world = ask_world(pixels)

    if len(pixels) != len(world) or len(pixels) < 4:
        raise SystemExit("need at least 4 matching pixel/world pairs")

    Hm, Hinv, resid = fit(pixels, world)
    poly = validity_polygon(pixels, args.dilate)

    print(f"\nreprojection residual per point (px): "
          f"{', '.join(f'{r:.2f}' for r in resid)}")
    print(f"max {resid.max():.2f} px", end="  ")
    if resid.max() > 2.0:
        print("<-- WARNING: above 2 px. The points are probably not coplanar, "
              "\n    or one was clicked on a kerb/skirting rather than the floor.")
    else:
        print("ok")

    f_solved, ortho = solve_focal(Hinv, W / 2.0, H_ / 2.0)
    if f_solved:
        if args.focal_px is None:
            args.focal_px = f_solved
        print(f"\nfocal length solved from the rectangle: {f_solved:.0f} px"
              f"   -> horizontal FOV {2*np.degrees(np.arctan(W/2/f_solved)):.1f} deg")
        print(f"  axis orthogonality residual {ortho:.4f}  (0 is perfect)", end="")
        print("   ok" if ortho < 0.10 else
              "\n  <-- WARNING: the recovered ground axes are not "
              "perpendicular.\n      The clicked quad is probably not the shape "
              "you said it was -- recheck\n      the tile counts, or the order "
              "the four points were clicked in.")
    else:
        print("\ncould not solve the focal length from this quad "
              "(degenerate geometry); pass --focal-px if you know it")

    name = args.camera or Path(args.video).stem
    out = Path(args.out or f"calib/{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "camera": name,
        "source_video": str(args.video),
        "frame": args.frame,
        "img_size": [W, H_],
        "max_side": args.max_side,
        "pixels": [list(map(float, p_)) for p_ in pixels],
        "world_m": [list(map(float, w)) for w in world],
        "H": Hm.tolist(),
        "H_inv": Hinv.tolist(),
        "residual_px": resid.tolist(),
        "residual_max_px": float(resid.max()),
        "validity_polygon": poly.tolist(),
        "validity_dilate": args.dilate,
        "focal_px": args.focal_px,
        "note": "H maps image pixels to ground metres (X, Y, Z=0). Scale is "
                "provisional: only the plane's shape is fixed here, absolute "
                "scale is re-fitted from the animal's proportions later.",
    }, indent=2))
    print(f"wrote {out}")

    if args.overlay:
        draw_overlay(img, Hm, Hinv, pixels, poly, resid, args.grid, args.overlay)


if __name__ == "__main__":
    main()
