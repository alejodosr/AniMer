"""Everything on one frame: SMAL mesh, calibrated floor, paws, and paw drift.

Built to answer one question -- why does a paw that is visibly on the floor get
labelled swing? Four layers, so the failure has nowhere to hide:

  * the fitted SMAL mesh, so you can see whether the body is where the dog is;
  * the calibrated ground plane as a grid, so you can see whether our idea of
    the floor matches the real floor;
  * the paw markers, filled when the pipeline says planted;
  * a TRAIL of each paw's position ON THE GROUND over the last N frames,
    projected back into the image.

The trail is the diagnostic. A genuinely planted paw should map to the same
spot on the floor every frame, so its trail should be a dot. If the paw looks
still in the video but its trail smears, the error is in the reconstruction,
not in the threshold -- and no amount of tuning fixes it.

Runs under the AniMer conda env (needs torch + pyrender), but NOT the 8.35 GB
checkpoint: the pose is already in the Phase B npz, so it only rebuilds the
SMAL layer from the model pickle.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np

LEGS = ["FR", "FL", "RR", "RL"]
LEG_BGR = {"FR": (60, 60, 235), "FL": (60, 170, 250),
           "RR": (80, 200, 80), "RL": (235, 150, 60)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer", required=True)
    p.add_argument("--contacts", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--animer-repo", default="/home/alejodosr/py_workspace/AniMer")
    p.add_argument("--trail", type=int, default=12, help="frames of ground trail")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--grid", type=float, default=0.5)
    args = p.parse_args()

    repo = Path(args.animer_repo).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import cv2
    import torch
    import trimesh
    import pyrender
    import pickle
    from amr.models.smal_warapper import SMALLayer

    b = np.load(args.infer, allow_pickle=True)
    c = np.load(args.contacts, allow_pickle=True)
    cal = json.loads(Path(args.calib).read_text())
    W, H = [int(v) for v in b["img_size"]]
    focal = float(b["focal_full"])
    fps = float(b["fps"])
    N = int(b["num_frames"])
    contacts = c["contacts"]
    ground = c["ground"]
    Hinv = np.array(cal["H_inv"])
    paw_src = c["paw_uv"] if "paw_uv" in c.files else b["paw_uv"]

    with open(repo / "data/smal/my_smpl_00781_4_all.pkl", "rb") as f:
        smal = SMALLayer(**pickle.load(f, encoding="latin1"))
    faces = smal.faces.numpy()

    print("re-running FK for the mesh", flush=True)
    verts = np.empty((N, 3889, 3), dtype=np.float32)
    with torch.no_grad():
        bt = torch.from_numpy(b["betas_frozen"]).float()[None]
        for s in range(0, N, 32):
            e = min(N, s + 32)
            o = SMALLayer.forward(
                smal, betas=bt.expand(e - s, -1),
                global_orient=torch.from_numpy(b["global_orient"][s:e]).float(),
                pose=torch.from_numpy(b["pose"][s:e]).float(), pose2rot=False)
            verts[s:e] = o.vertices.numpy()

    flip = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    renderer = pyrender.OffscreenRenderer(W, H, point_size=1.0)
    cam = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=W / 2., cy=H / 2.,
                                    zfar=1e6)
    mat = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, roughnessFactor=0.75, alphaMode="OPAQUE",
        baseColorFactor=(0.35, 0.72, 0.95, 1.0))

    def ground_to_img(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        h = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ Hinv.T
        return h[:, :2] / h[:, 2:3]

    gl = ground[np.isfinite(ground).all(-1)]
    lo, hi = gl.min(0) - 1.0, gl.max(0) + 1.0

    cap = cv2.VideoCapture(args.video)
    writer = None
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok or idx + 1 >= N:
            break
        idx += 1
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

        # --- calibrated floor grid ---
        for x in np.arange(np.floor(lo[0]), hi[0], args.grid):
            a, bb = ground_to_img([[x, lo[1]], [x, hi[1]]])
            cv2.line(frame, tuple(np.int32(a)), tuple(np.int32(bb)),
                     (0, 190, 235), 1, cv2.LINE_AA)
        for y in np.arange(np.floor(lo[1]), hi[1], args.grid):
            a, bb = ground_to_img([[lo[0], y], [hi[0], y]])
            cv2.line(frame, tuple(np.int32(a)), tuple(np.int32(bb)),
                     (0, 190, 235), 1, cv2.LINE_AA)

        # --- SMAL mesh ---
        v = verts[idx] + b["cam_t"][idx]
        mesh = trimesh.Trimesh(v, faces, process=False)
        mesh.apply_transform(flip)
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(.4, .4, .4))
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=mat))
        node = pyrender.Node(camera=cam, matrix=np.eye(4))
        scene.add_node(node)
        for pos in [(0, -1, 1), (1, 1, 2), (-1, 1, 2)]:
            lp = np.eye(4)
            lp[:3, 3] = pos
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5),
                      pose=lp)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        rgba = color.astype(np.float32) / 255.0
        m = rgba[:, :, 3:4] * args.alpha
        frame = (rgba[:, :, :3][:, :, ::-1] * 255.0 * m
                 + frame.astype(np.float32) * (1 - m)).astype(np.uint8)

        # --- paw ground trails, projected back into the image ---
        s0 = max(0, idx - args.trail)
        for li, leg in enumerate(LEGS):
            tr = ground[s0:idx + 1, li]
            tr = tr[np.isfinite(tr).all(-1)]
            if len(tr) > 1:
                uv = np.int32(ground_to_img(tr))
                for k in range(len(uv) - 1):
                    cv2.line(frame, tuple(uv[k]), tuple(uv[k + 1]),
                             LEG_BGR[leg], 2, cv2.LINE_AA)

        # --- paw markers ---
        for li, leg in enumerate(LEGS):
            u, v_ = paw_src[idx, li]
            down = bool(contacts[idx, li])
            col = LEG_BGR[leg]
            cv2.circle(frame, (int(round(u)), int(round(v_))), 10 if down else 7,
                       col, -1 if down else 2, cv2.LINE_AA)
            if down:
                cv2.circle(frame, (int(round(u)), int(round(v_))), 10,
                           (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, leg, (int(u) + 12, int(v_) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        cv2.putText(frame, f"t={idx/fps:5.2f}s   {int(contacts[idx].sum())} planted"
                    f"   filled = planted   coloured trail = paw position ON THE "
                    f"FLOOR, last {args.trail} frames",
                    (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)

        if writer is None:
            writer = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                 "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}",
                 "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "medium",
                 "-crf", "18", "-pix_fmt", "yuv420p", str(args.out)],
                stdin=subprocess.PIPE)
        writer.stdin.write(np.ascontiguousarray(frame).tobytes())
        if idx % 48 == 0:
            print(f"  {idx}/{N}", flush=True)

    cap.release()
    renderer.delete()
    if writer:
        writer.stdin.close()
        writer.wait()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
