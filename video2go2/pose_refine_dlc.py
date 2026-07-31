"""Refine AniMer's SMAL pose against DLC's 2D skeleton (SMPLify-style).

WHY THIS AND NOT MORE PAW WORK. On the harness, with PERFECT paw pixels the
root error floor was still 0.034 m — the limit is AniMer's 3D body fit, not
the paw pixels. DLC observes the WHOLE skeleton independently, so the leverage
is in correcting the body, not in polishing one landmark.

CORRESPONDENCE (established empirically, not guessed — see
smal_dlc_correspondence.npz). DLC's naming is anatomically loose and taking it
at face value would have wired the skeleton wrong:

    DLC front_*_thai  -> elbow        DLC back_*_thai  -> stifle (true knee)
    DLC front_*_knee  -> WRIST/carpus DLC back_*_knee  -> HOCK/ankle
    DLC *_paw         -> paw

Those twelve leg pairs agree with AniMer's own projected skeleton to 12-24 px,
which is an independent cross-check on both models. Head landmarks are NOT
matched geometrically (they cluster within ~25 px and the assignment scrambles
them); they are pinned by name and down-weighted.

WHAT IS OPTIMISED. Per-frame joint rotations in 6D, initialised from AniMer
and held near it by a prior, plus the per-frame camera translation. Betas stay
frozen (the whole pipeline depends on a constant skeleton). Residuals:
reprojection to DLC keypoints weighted by DLC confidence, a prior pulling
pose toward AniMer's estimate, and temporal smoothness.

The output is a drop-in replacement for the Phase B npz, so contacts_kine and
world_place_ba consume it unchanged.
"""
from pathlib import Path
import argparse
import pickle
import sys

import numpy as np

# SMAL-26 landmark index -> DLC keypoint name, from the empirical matching.
LEG_PAIRS = {
    3: "front_left_paw",   4: "front_right_paw",
    5: "back_left_paw",    6: "back_right_paw",
    8: "front_left_thai",  9: "front_right_thai",     # elbows
    14: "front_left_knee", 15: "front_right_knee",    # wrists
    10: "back_left_thai",  11: "back_right_thai",     # stifles
    16: "back_left_knee",  17: "back_right_knee",     # hocks
}
TRUNK_PAIRS = {7: "back_end", 25: "tail_base", 19: "tail_end", 18: "back_base"}
HEAD_PAIRS = {24: "nose", 2: "lower_jaw", 22: "left_eye", 23: "right_eye",
              0: "left_earbase", 1: "right_earbase"}


def rot6d_to_matrix(x):
    import torch
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(R):
    import torch
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer", required=True)
    p.add_argument("--dlc", required=True, help="npz with kp (N,39,3) and names")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default="/home/alejodosr/py_workspace/AniMer")
    p.add_argument("--src-size", default="3360,2100",
                   help="resolution DLC ran at, for coordinate rescaling")
    p.add_argument("--iters", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--w-leg", type=float, default=1.0)
    p.add_argument("--w-trunk", type=float, default=1.0)
    p.add_argument("--w-head", type=float, default=0.25)
    p.add_argument("--w-prior", type=float, default=6.0,
                   help="pull toward AniMer's pose; the DLC signal is 2D only "
                        "so the depth direction must come from the prior")
    p.add_argument("--w-smooth", type=float, default=12.0)
    p.add_argument("--min-conf", type=float, default=0.4)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import torch
    sys.path.insert(0, args.repo)
    from amr.models.smal_warapper import SMALLayer, keypoint_vertices_idx

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    b = np.load(args.infer, allow_pickle=True)
    D = np.load(args.dlc, allow_pickle=True)
    names = [str(x) for x in D["names"]]
    dlc = D["kp"]

    W, H = [int(v) for v in b["img_size"]]
    sw, sh = [int(v) for v in args.src_size.split(",")]
    sx, sy = W / sw, H / sh
    F = float(b["focal_full"])
    N = min(len(dlc), int(b["num_frames"]))
    if args.max_frames:
        N = min(N, args.max_frames)

    # ---- assemble observations -------------------------------------------
    obs_lm, obs_col, obs_w = [], [], []
    for table, w in ((LEG_PAIRS, args.w_leg), (TRUNK_PAIRS, args.w_trunk),
                     (HEAD_PAIRS, args.w_head)):
        for lm, nm in table.items():
            if nm in names:
                obs_lm.append(lm)
                obs_col.append(names.index(nm))
                obs_w.append(w)
    obs_lm = np.array(obs_lm)
    uv = dlc[:N][:, obs_col, :2] * np.array([sx, sy])
    cf = dlc[:N][:, obs_col, 2]
    wt = (cf * (cf > args.min_conf)) * np.array(obs_w)[None, :]
    print(f"{len(obs_lm)} landmark correspondences; "
          f"{100 * (cf > args.min_conf).mean():.0f}% of observations above "
          f"confidence {args.min_conf}")

    smal = SMALLayer(**pickle.load(open(
        Path(args.repo) / "data/smal/my_smpl_00781_4_all.pkl", "rb"),
        encoding="latin1")).to(dev)
    for prm in smal.parameters():
        prm.requires_grad_(False)

    go0 = torch.from_numpy(b["global_orient"][:N]).float().to(dev)   # (N,1,3,3)
    po0 = torch.from_numpy(b["pose"][:N]).float().to(dev)            # (N,34,3,3)
    ct0 = torch.from_numpy(b["cam_t"][:N]).float().to(dev)
    betas = torch.from_numpy(b["betas_frozen"]).float()[None].to(dev)
    uv_t = torch.from_numpy(uv).float().to(dev)
    wt_t = torch.from_numpy(wt).float().to(dev)

    g6 = matrix_to_rot6d(go0).clone().requires_grad_(True)
    p6 = matrix_to_rot6d(po0).clone().requires_grad_(True)
    ct = ct0.clone().requires_grad_(True)
    g6_0, p6_0 = g6.detach().clone(), p6.detach().clone()

    groups = [torch.tensor(ix, device=dev) for ix in
              [keypoint_vertices_idx[i] for i in obs_lm]]
    opt = torch.optim.Adam([g6, p6, ct], lr=args.lr)

    def forward():
        o = SMALLayer.forward(smal, betas=betas.expand(N, -1),
                              global_orient=rot6d_to_matrix(g6),
                              pose=rot6d_to_matrix(p6), pose2rot=False)
        v = o.vertices + ct[:, None, :]
        lm = torch.stack([v[:, g, :].mean(1) for g in groups], dim=1)
        z = lm[..., 2].clamp(min=1e-3)
        return torch.stack([F * lm[..., 0] / z + W / 2.0,
                            F * lm[..., 1] / z + H / 2.0], dim=-1)

    print("optimising", flush=True)
    for it in range(args.iters):
        opt.zero_grad()
        proj = forward()
        rep = ((proj - uv_t).norm(dim=-1) * wt_t).sum() / wt_t.sum().clamp(min=1)
        prior = ((g6 - g6_0) ** 2).mean() + ((p6 - p6_0) ** 2).mean()
        sm = ((p6[1:] - p6[:-1]) ** 2).mean() + ((g6[1:] - g6[:-1]) ** 2).mean()
        loss = rep + args.w_prior * 1e3 * prior + args.w_smooth * 1e3 * sm
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == args.iters - 1:
            print(f"  it {it:4d}  reproj {rep.item():6.2f} px   "
                  f"prior {prior.item():.2e}  smooth {sm.item():.2e}", flush=True)

    with torch.no_grad():
        proj0 = None
        go_n = rot6d_to_matrix(g6).cpu().numpy()
        po_n = rot6d_to_matrix(p6).cpu().numpy()
        ct_n = ct.cpu().numpy()
        o = SMALLayer.forward(smal, betas=betas.expand(N, -1),
                              global_orient=rot6d_to_matrix(g6),
                              pose=rot6d_to_matrix(p6), pose2rot=False)
        joints = o.joints.cpu().numpy()
        verts = o.vertices.cpu().numpy()

    FK_ROWS = [0, 6, 11, 7, 21, 17]
    PAW_KP = [4, 3, 6, 5]
    paws = np.stack([verts[:, keypoint_vertices_idx[k], :].mean(1) for k in PAW_KP], 1)
    pts = np.concatenate([joints[:, FK_ROWS, :], paws], axis=1)
    root = pts[:, 0:1, :].copy()
    paw_cam = paws + ct_n[:, None, :]
    zc = np.maximum(paw_cam[..., 2], 1e-6)
    paw_uv = np.stack([F * paw_cam[..., 0] / zc + W / 2.0,
                       F * paw_cam[..., 1] / zc + H / 2.0], -1)

    out = {k: b[k] for k in b.files}
    out.update(dict(num_frames=N, global_orient=go_n.astype(np.float64),
                    pose=po_n.astype(np.float64), cam_t=ct_n.astype(np.float64),
                    points_local=(pts - root).astype(np.float64),
                    root_model=root[:, 0, :].astype(np.float64),
                    paw_uv=paw_uv.astype(np.float64),
                    valid=b["valid"][:N], frame_idx=b["frame_idx"][:N],
                    refined_with="dlc_skeleton"))
    o_ = Path(args.out)
    o_.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o_, **out)
    print(f"wrote {o_}")


if __name__ == "__main__":
    main()
