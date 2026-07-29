"""Phase 0 (PLAN.md) — synthetic harness with exact ground truth.

Takes a real recovered clip (dog_1's world motion) as the TRUE motion, re-films
it from a grid of synthetic cameras (height x distance), and fabricates the
AniMer-style inference npz each camera would produce, with a controlled noise
model. Every downstream quantity — toe world positions, contacts, scale,
camera pose — is known exactly, so the pipeline's error can finally be
measured rather than inferred.

Two claims this exists to test:
  1. the baseline (contacts_ground + world_place) floor error scales like
     distance / camera-height, as STATUS.md predicts;
  2. later, that world_place_ba.py breaks that scaling.

Noise is injected into the FIT, not into derived arrays: global_orient and
pose get band-limited (0.5-3 Hz, the measured error band) rotation drift, and
cam_t gets pixel + relative-depth drift. points_local and paw_uv are then
recomputed from the perturbed fit exactly the way animer_infer.py computes
them, so contacts_ground.refine_paw_pixels — which re-runs FK from the stored
pose — sees the SAME perturbed body as everything else. Perturbing the output
arrays instead would let that re-FK silently launder the noise back out.

The same noise realization (same seed, applied in the source body frame) is
shared by every camera config, so the sweep isolates geometry: any difference
between configs is the camera, not the dice.

Fabrication of cam_t honours AniMer's weak-perspective convention: the stored
focal_full is NOT the true focal (Traps: never use it geometrically), so
cam_t_z is scaled by focal_full/f_true and cam_t_xy solved so the root lands
on its true pixel. The paws then reproject with genuine weak-perspective
distortion, like the real thing.

Outputs, under --outdir:
    gt_motion.npz              shared truth: world points, GT contacts, scale
    cfg_h<h>_d<d>/gt.npz       per-config camera truth + amplification factor
    cfg_h<h>_d<d>/animer.npz   fabricated inference npz (pipeline input)
    cfg_h<h>_d<d>/calib.json   via the real calibrate_ground fit/solve_focal
    cfg_h<h>_d<d>/render.mp4   optional (--render), for full-AniMer runs
"""
from pathlib import Path
import argparse
import json
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_ground import fit as fit_homography, solve_focal, validity_polygon
from contacts_2d import runs

# points_local layout (animer_infer.POINT_NAMES): 0 root, 1 chest,
# 2-5 mounts FR FL RR RL, 6-9 toes FR FL RR RL
TOE0 = 6
# canonical joint rows and paw vertex groups, copied from animer_infer.py
FK_ROWS = [0, 6, 11, 7, 21, 17]           # root, chest, mounts FR FL RR RL
PAW_KP = [4, 3, 6, 5]                     # FR FL RR RL (front pair transposed)


# ---------------------------------------------------------------- SMAL FK ---
def load_smal(repo):
    import pickle
    import torch  # noqa: F401  (SMALLayer needs it imported)
    sys.path.insert(0, str(repo))
    from amr.models.smal_warapper import SMALLayer
    with open(Path(repo) / "data/smal/my_smpl_00781_4_all.pkl", "rb") as f:
        return SMALLayer(**pickle.load(f, encoding="latin1"))


def fk(smal, betas, go, pose, batch=128):
    """verts (N,3889,3), 10 canonical points (N,10,3) — model units, cam frame."""
    import torch
    from amr.models.smal_warapper import keypoint_vertices_idx
    N = len(go)
    verts = np.empty((N, 3889, 3), np.float64)
    joints = np.empty((N, 35, 3), np.float64)
    with torch.no_grad():
        bt = torch.from_numpy(betas).float()[None]
        for s in range(0, N, batch):
            e = min(N, s + batch)
            o = type(smal).forward(  # SMALLayer.forward: rigid joints, not landmarks
                smal, betas=bt.expand(e - s, -1),
                global_orient=torch.from_numpy(go[s:e]).float(),
                pose=torch.from_numpy(pose[s:e]).float(), pose2rot=False)
            verts[s:e] = o.vertices.numpy()
            joints[s:e] = o.joints.numpy()
    paws = np.stack([verts[:, keypoint_vertices_idx[k], :].mean(axis=1)
                     for k in PAW_KP], axis=1)
    points = np.concatenate([joints[:, FK_ROWS, :], paws], axis=1)
    return verts, points


# ------------------------------------------------------------------ noise ---
def bandlimited(rng, shape, fps, sigma, lo=0.5, hi=3.0):
    """Band-limited Gaussian drift, std sigma per channel. shape[0] is time."""
    from scipy.signal import butter, filtfilt
    x = rng.standard_normal(shape)
    b, a = butter(2, [lo / (0.5 * fps), min(hi / (0.5 * fps), 0.99)],
                  btype="band")
    flat = filtfilt(b, a, x.reshape(len(x), -1), axis=0)
    std = flat.std(axis=0, keepdims=True)
    flat = flat / np.maximum(std, 1e-12) * sigma
    return flat.reshape(shape)


def rot_drift(rng, n, j, fps, sigma_deg):
    """(n, j, 3, 3) rotation noise: band-limited rotvec of std sigma_deg."""
    from scipy.spatial.transform import Rotation
    w = bandlimited(rng, (n, j, 3), fps, np.deg2rad(sigma_deg))
    return Rotation.from_rotvec(w.reshape(-1, 3)).as_matrix().reshape(n, j, 3, 3)


# ----------------------------------------------------------------- camera ---
def make_camera(traj_xyz, ctr2, perp2, d, h, W, H, margin=0.88):
    """Static camera at horizontal distance d, height h, perpendicular to the
    trajectory's principal axis, focal auto-fitted so the motion stays in
    frame. Returns R_wc (world->cam, x right / y down / z forward), C, f."""
    C = np.array([ctr2[0] + perp2[0] * d, ctr2[1] + perp2[1] * d, h])
    look = np.array([ctr2[0], ctr2[1], 0.35])
    f_axis = look - C
    f_axis = f_axis / np.linalg.norm(f_axis)
    right = np.cross(f_axis, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f_axis, right)
    R_wc = np.stack([right, down, f_axis], axis=0)

    pts = traj_xyz.reshape(-1, 3)
    cam = (pts - C) @ R_wc.T
    z = cam[:, 2]
    if (z <= 0.05).any():
        raise SystemExit(f"camera d={d} h={h}: body passes behind the camera")
    fu = margin * (W / 2.0) / np.abs(cam[:, 0] / z).max()
    fv = margin * (H / 2.0) / np.abs(cam[:, 1] / z).max()
    f = float(np.clip(min(fu, fv), 350.0, 2600.0))
    cov = np.mean((np.abs(f * cam[:, 0] / z) < W / 2) &
                  (np.abs(f * cam[:, 1] / z) < H / 2))
    return R_wc, C, f, float(cov)


def project(K_f, W, H, cam_pts):
    z = np.maximum(cam_pts[..., 2], 1e-9)
    return np.stack([K_f * cam_pts[..., 0] / z + W / 2.0,
                     K_f * cam_pts[..., 1] / z + H / 2.0], axis=-1)


# ------------------------------------------------------------------ render --
def render_config(cfg_dir, verts_world, faces, R_wc, C, f, W, H, fps):
    """pyrender EGL: checkerboard floor + dog mesh, one mp4 per config."""
    import os
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import cv2
    import pyrender
    import trimesh

    lo = verts_world.reshape(-1, 3).min(axis=0)[:2] - 6.0
    hi = verts_world.reshape(-1, 3).max(axis=0)[:2] + 6.0
    # floor: checkerboard texture, 0.5 m squares
    n = 24
    tex = np.zeros((n * 16, n * 16, 3), np.uint8)
    for i in range(n):
        for j in range(n):
            c = np.array([196, 189, 178]) if (i + j) % 2 else np.array([124, 118, 108])
            tex[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16] = c
    sq = 0.5
    ext_x, ext_y = n * sq, n * sq
    cx, cy = (lo + hi) / 2.0
    fv = np.array([[cx - ext_x / 2, cy - ext_y / 2, 0], [cx + ext_x / 2, cy - ext_y / 2, 0],
                   [cx + ext_x / 2, cy + ext_y / 2, 0], [cx - ext_x / 2, cy + ext_y / 2, 0]])
    floor = trimesh.Trimesh(vertices=fv, faces=[[0, 1, 2], [0, 2, 3]],
                            visual=trimesh.visual.TextureVisuals(
                                uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float),
                                image=__import__("PIL.Image", fromlist=["Image"]).fromarray(tex)),
                            process=False)

    scene = pyrender.Scene(bg_color=[0.72, 0.78, 0.84, 1.0],
                           ambient_light=[0.45, 0.45, 0.45])
    scene.add(pyrender.Mesh.from_trimesh(floor, smooth=False))
    key = pyrender.DirectionalLight(color=np.ones(3), intensity=3.2)
    pose_l = np.eye(4)
    pose_l[:3, :3] = trimesh.transformations.euler_matrix(-0.9, 0.35, 0)[:3, :3]
    scene.add(key, pose=pose_l)

    # camera: pyrender is OpenGL (-z forward, +y up); ours is CV (+z fwd, +y down)
    cam_pose = np.eye(4)
    cam_pose[:3, 0] = R_wc[0]
    cam_pose[:3, 1] = -R_wc[1]
    cam_pose[:3, 2] = -R_wc[2]
    cam_pose[:3, 3] = C
    scene.add(pyrender.IntrinsicsCamera(fx=f, fy=f, cx=W / 2.0, cy=H / 2.0,
                                        znear=0.05, zfar=80.0), pose=cam_pose)

    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.48, 0.33, 0.22, 1.0], roughnessFactor=0.85)
    r = pyrender.OffscreenRenderer(W, H)
    vw = cv2.VideoWriter(str(cfg_dir / "render.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    node = None
    for t in range(len(verts_world)):
        tm = trimesh.Trimesh(verts_world[t], faces, process=False)
        mesh = pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True)
        if node is not None:
            scene.remove_node(node)
        node = scene.add(mesh)
        color, _ = r.render(scene)
        vw.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        if t == 0:
            cv2.imwrite(str(cfg_dir / "render_f0.png"),
                        cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    vw.release()
    r.delete()
    print(f"    rendered {len(verts_world)} frames -> {cfg_dir / 'render.mp4'}")


# ------------------------------------------------------------------- main ---
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer", default="/media/SHARED_DATA/postcapitalistrobots/"
                   "animer/v2go2/dog_1_animer.npz")
    p.add_argument("--world", default="/media/SHARED_DATA/postcapitalistrobots/"
                   "animer/v2go2/dog_1_world.npz")
    p.add_argument("--repo", default="/home/alejodosr/py_workspace/AniMer")
    p.add_argument("--outdir", default="/media/SHARED_DATA/postcapitalistrobots/"
                   "animer/v2go2/synth")
    p.add_argument("--trim", default="0,660",
                   help="source frames to keep; dog_1's tail is unusable")
    p.add_argument("--heights", default="0.8,1.1,1.6,2.2")
    p.add_argument("--distances", default="2,4,6")
    p.add_argument("--img", default="1280,720")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--pose-noise-deg", type=float, default=1.5)
    p.add_argument("--orient-noise-deg", type=float, default=1.0)
    p.add_argument("--pix-noise-px", type=float, default=2.5,
                   help="band-limited drift of the root pixel (localisation)")
    p.add_argument("--depth-noise-frac", type=float, default=0.025,
                   help="band-limited relative drift of the fitted depth")
    p.add_argument("--focal-full", type=float, default=5000.0,
                   help="the assumed focal the fabricated cam_t is expressed "
                        "in, matching AniMer's convention")
    p.add_argument("--render", default="",
                   help="'all' or comma list like '1.1x4,2.2x2' of hxd configs")
    p.add_argument("--clean", action="store_true",
                   help="delete and rebuild --outdir")
    args = p.parse_args()

    a, b = [int(x) for x in args.trim.split(",")]
    sl = slice(a, b)
    W, H = [int(x) for x in args.img.split(",")]
    heights = [float(x) for x in args.heights.split(",")]
    dists = [float(x) for x in args.distances.split(",")]

    src = np.load(args.infer, allow_pickle=True)
    wld = np.load(args.world, allow_pickle=True)
    fps = float(src["fps"])
    mpu = float(wld["metres_per_unit"])
    R_cw_src = wld["R_cw"]
    go = src["global_orient"][sl]
    pose = src["pose"][sl]
    betas = src["betas_frozen"]
    pts_local_stored = src["points_local"][sl]
    world_pts = wld["world"][sl]                     # (N,10,3) TRUE motion, metres
    T = world_pts[:, 0, :]                           # true root position
    N = len(go)

    out = Path(args.outdir)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- clean FK: must reproduce the stored points_local ------------------
    print(f"source motion: {N} frames at {fps:.1f} fps, scale {mpu} m/unit")
    smal = load_smal(args.repo)
    faces = smal.faces.numpy()
    verts_c, pts_c = fk(smal, betas, go, pose)
    root_c = pts_c[:, 0:1, :]
    repro = np.abs((pts_c - root_c) - pts_local_stored).max()
    print(f"  FK reproduction vs stored points_local: max |diff| {repro:.2e} "
          f"(must be ~0)")
    if repro > 1e-3:
        raise SystemExit("FK does not reproduce the source npz -- wrong "
                         "checkpoint pkl or wrong point extraction")

    # TRUE world vertices for rendering, and true toe check
    verts_w = np.einsum("ij,nvj->nvi", R_cw_src,
                        (verts_c - root_c) * mpu) + T[:, None, :]
    toe_chk = np.einsum("ij,nkj->nki", R_cw_src,
                        pts_local_stored[:, TOE0:] * mpu) + T[:, None, :]
    assert np.abs(toe_chk - world_pts[:, TOE0:]).max() < 1e-9

    # ---- ground-truth contacts -------------------------------------------
    # DECLARED, not derived: the source pipeline's own labels. The recovered
    # dog_1 motion carries the very mesh drift this harness studies, so its
    # stance toes move at ~0.3 m/s median and no kinematic threshold
    # reproduces a plausible duty from it. The labels are identical for every
    # config, so cross-config comparisons are unaffected; the wander below is
    # the known fuzziness bound on anything scored against these labels.
    gt_contacts = wld["contacts"][sl]
    toe_w = world_pts[:, TOE0:, :]
    wander = []
    for leg in range(4):
        for s_, e_, v_ in runs(gt_contacts[:, leg]):
            if v_ and e_ - s_ >= 3:
                t_ = toe_w[s_:e_, leg, :2]
                wander.append(np.linalg.norm(t_ - np.median(t_, axis=0),
                                             axis=1).max())
    wander_med = float(np.median(wander)) if wander else np.nan
    print(f"  GT contacts (declared from source): duty {gt_contacts.mean():.2f}"
          f", GT stance-toe wander median {wander_med * 100:.1f} cm "
          f"(fuzziness bound)")

    # ---- ONE shared noise realization, in the source body frame ------------
    rng = np.random.default_rng(args.seed)
    go_n = rot_drift(rng, N, 1, fps, args.orient_noise_deg) @ go
    pose_n = rot_drift(rng, N, 34, fps, args.pose_noise_deg) @ pose
    pix_drift = bandlimited(rng, (N, 2), fps, args.pix_noise_px)
    depth_drift = bandlimited(rng, (N, 1), fps, args.depth_noise_frac)[:, 0]

    verts_n, pts_n = fk(smal, betas, go_n, pose_n)
    root_n = pts_n[:, 0:1, :]
    pts_local_n = pts_n - root_n                      # noisy fit, model units

    # Rest root joint j0: the pivot LBS rotates global_orient about. Needed
    # because cam_t pairs with the RAW FK frame (AniMer's convention -- see
    # animer_infer paw_uv), and raw FK transforms as R@(x - j0) + j0, not R@x.
    # Getting this wrong shifted every fabricated observation by a constant
    # ~0.3 m and cost a debugging session; hence the assertion below.
    import torch as _t
    with _t.no_grad():
        v_shaped = smal.v_template + _t.einsum(
            "vcb,b->vc", smal.shapedirs, _t.from_numpy(betas).float())
        j0 = (smal.J_regressor[0:1] @ v_shaped).numpy()[0]

    # how big is the injected error, in metres, where it matters?
    paw_err = np.linalg.norm((pts_local_n - pts_local_stored)[:, TOE0:], axis=-1) * mpu
    print(f"  injected 3D paw error: median {np.median(paw_err) * 100:.1f} cm, "
          f"p90 {np.percentile(paw_err, 90) * 100:.1f} cm  "
          f"(target: the measured 2-5 cm)")

    np.savez_compressed(out / "gt_motion.npz",
                        source=str(src["source"]), fps=fps, trim=[a, b],
                        world=world_pts, contacts=gt_contacts,
                        metres_per_unit=mpu,
                        gt_stance_wander_med_m=wander_med,
                        paw_err_med_m=float(np.median(paw_err)),
                        seed=args.seed)

    # focal auto-fit must see the whole MESH, not just the skeleton points —
    # fitting to the 10 points crops the head and ears out of the frame
    fit_pts = np.concatenate([world_pts, verts_w[:, ::37]], axis=1)

    # trajectory frame: walk axis + its normal (the camera side is +perp2)
    root2 = world_pts[:, 0, :2]
    ctr2 = root2.mean(axis=0)
    _, _, Vt = np.linalg.svd(root2 - ctr2, full_matrices=False)
    axis2 = Vt[0]
    perp2 = np.array([-axis2[1], axis2[0]])
    all2 = world_pts[..., :2].reshape(-1, 2) - ctr2
    proj_a, proj_p = all2 @ axis2, all2 @ perp2
    lo_a, hi_a = proj_a.min() - 0.8, proj_a.max() + 0.8

    render_set = set()
    if args.render == "all":
        render_set = {(h_, d_) for h_ in heights for d_ in dists}
    elif args.render:
        for tok in args.render.split(","):
            h_, d_ = tok.split("x")
            render_set.add((float(h_), float(d_)))

    print(f"\n{'config':>14} {'f_px':>6} {'f_solved':>8} {'amp d/h':>8} "
          f"{'cover':>6} {'pawpx_err':>9}")
    rows = []
    for h_ in heights:
        for d_ in dists:
            name = f"cfg_h{h_:g}_d{d_:g}"
            cfg = out / name
            cfg.mkdir(exist_ok=True)
            R_wc, C, f_true, cov = make_camera(fit_pts, ctr2, perp2,
                                               d_, h_, W, H)

            # calibration rectangle: spread over the VISIBLE floor, the way a
            # person would click — from just inside the bottom of the frame
            # out past the far side of the trajectory. A rectangle hugging
            # the long, narrow trajectory bbox projects to a nearly flat quad
            # at low camera heights and calibrate_ground's degeneracy guard
            # rejects it (correctly: real clicks like that would too).
            depression = float(np.arcsin(np.clip(-R_wc[2, 2], -1, 1)))
            bot = depression + np.arctan(0.93 * (H / 2.0) / f_true)
            x_vis = h_ / np.tan(bot) if bot > 1e-3 else 0.3
            near_p = d_ - max(x_vis + 0.15, 0.6)
            near_p = max(near_p, proj_p.max() + 0.3)
            far_p = proj_p.min() - 0.6
            rect_w = np.array([ctr2 + s * axis2 + t * perp2
                               for s, t in [(lo_a, near_p), (hi_a, near_p),
                                            (hi_a, far_p), (lo_a, far_p)]])
            e1, e2 = rect_w[1] - rect_w[0], rect_w[3] - rect_w[0]
            if e1[0] * e2[1] - e1[1] * e2[0] < 0:    # world X x Y must be +Z
                rect_w = rect_w[[0, 3, 2, 1]]

            # --- fabricated cam_t, weak-perspective convention --------------
            P_cam = (T - C) @ R_wc.T                       # true root, cam frame
            u_true = project(f_true, W, H, P_cam)
            u_root = u_true + pix_drift
            depth = P_cam[:, 2] * (1.0 + depth_drift)
            ct_z = depth * (args.focal_full / f_true) / mpu
            ct = np.stack([(u_root[:, 0] - W / 2) * ct_z / args.focal_full,
                           (u_root[:, 1] - H / 2) * ct_z / args.focal_full,
                           ct_z], axis=1)

            # --- noisy body, rotated into this camera's orientation ---------
            R_delta = R_wc @ R_cw_src
            pl = np.einsum("ij,nkj->nki", R_delta, pts_local_n)
            go_cfg = R_delta[None, None] @ go_n              # (N,1,3,3)
            paw_uv = project(args.focal_full, W, H,
                             pl[:, TOE0:] + ct[:, None, :])
            # cam_t in the raw-FK convention consumed by refine_paw_pixels:
            # raw root of FK(R_delta@go) is R_delta@(root - j0) + j0
            root_raw = (root_n[:, 0, :] - j0) @ R_delta.T + j0
            ct_raw = ct - root_raw
            if not rows:      # first config: verify the identity vs real FK
                _, pts_chk = fk(smal, betas, go_cfg[:4], pose_n[:4])
                err_raw = np.abs(pts_chk[:, 0] - root_raw[:4]).max()
                if err_raw > 1e-4:
                    raise SystemExit(f"raw-root identity broken: {err_raw:.2e}")

            # honest pixel error: fabricated paw pixels vs TRUE paw pixels
            true_paw_uv = project(f_true, W, H,
                                  (world_pts[:, TOE0:] - C) @ R_wc.T)
            px_err = float(np.median(np.linalg.norm(paw_uv - true_paw_uv,
                                                    axis=-1)))

            np.savez_compressed(
                cfg / "animer.npz",
                source=name, fps=fps, num_frames=N,
                frame_idx=np.arange(N), valid=np.ones(N, bool),
                points_local=pl, root_model=root_raw,
                paw_uv=paw_uv, global_orient=go_cfg, pose=pose_n,
                betas_frozen=betas, cam_t=ct_raw,
                img_size=np.array([W, H]), focal_full=float(args.focal_full),
                detection_rate=1.0)

            # --- calibration through the real code path ---------------------
            rect_px = project(f_true, W, H,
                              (np.concatenate([rect_w, np.zeros((4, 1))], 1) - C)
                              @ R_wc.T)
            Hm, Hinv, resid = fit_homography(rect_px.tolist(), rect_w.tolist())
            f_solved, ortho = solve_focal(Hinv, W / 2.0, H / 2.0)
            poly = validity_polygon(rect_px.tolist())
            (cfg / "calib.json").write_text(json.dumps({
                "camera": name, "source_video": "synthetic", "frame": 0,
                "img_size": [W, H], "max_side": 1280,
                "pixels": rect_px.tolist(), "world_m": rect_w.tolist(),
                "H": Hm.tolist(), "H_inv": Hinv.tolist(),
                "residual_px": resid.tolist(),
                "residual_max_px": float(resid.max()),
                "validity_polygon": poly.tolist(), "validity_dilate": 1.25,
                "focal_px": f_solved,
                "note": "synthetic ground-truth camera (synth_harness.py)"},
                indent=2))

            amp = float(np.mean(np.linalg.norm((T - C)[:, :2], axis=1)) / h_)
            np.savez_compressed(cfg / "gt.npz",
                                R_wc=R_wc, C=C, f_true=f_true,
                                f_solved=f_solved if f_solved else np.nan,
                                img_size=np.array([W, H]),
                                amplification=amp, coverage=cov,
                                height=h_, distance=d_,
                                paw_px_err_med=px_err)
            rows.append((name, f_true, f_solved or float("nan"), amp, cov, px_err))
            print(f"{name:>14} {f_true:6.0f} {f_solved or float('nan'):8.1f} "
                  f"{amp:8.2f} {cov:6.2f} {px_err:9.2f}")

            if (h_, d_) in render_set:
                render_config(cfg, verts_w, faces, R_wc, C, f_true, W, H, fps)

    print(f"\nwrote {len(rows)} configs under {out}")


if __name__ == "__main__":
    main()
