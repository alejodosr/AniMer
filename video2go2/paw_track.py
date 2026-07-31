"""Paw pixel tracks from a point tracker, independent of the SMAL mesh fit.

The pipeline's limiting error is that the mesh-projected paw wanders 1-3 Hz
even when the paw is provably still. Measured on dog_1 over 280 paw-frames in
windows where the RAW PIXELS do not change (so any reported motion is pure
error): the mesh moves 3.45 px, CoTracker3 moves 0.02 px.

Two implementation points that decide whether this works at all:

  * NATIVE-RESOLUTION CROPS. CoTracker's predictor resizes its input to
    interp_shape = (384, 512). Handing it the 1280x800 frame throws away a
    factor of 2.5 and caps precision near the mesh's own error. Cropping
    512x384 around the paw means the resize is the identity.

  * CHAINED SEEDING WITH A MESH LEASH. Only the first chunk is seeded from
    the mesh; each later chunk is seeded from the previous chunk's own
    tracked position, so mesh jitter is not re-imported every chunk. But a
    tracker that loses its point stays lost, so the track is re-seeded from
    the mesh whenever it drifts further than --leash px away. That bounds the
    failure without importing the noise (leash >> mesh noise).

Output npz has the same `paw_uv` contract the rest of the pipeline consumes,
plus per-frame visibility, so contacts_kine.py and world_place_ba.py can use
it in place of the mesh projection.

Licence note: CoTracker is released by Meta under CC-BY-NC 4.0. Fine for
research/evaluation; for anything commercial, BootsTAPIR (Apache-2.0) is the
drop-in substitute (--backend bootstap, not yet implemented).
"""
from pathlib import Path
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

LEGS = ["FR", "FL", "RR", "RL"]
CW, CH = 512, 384          # must equal the predictor's interp_shape (h=384,w=512)


def read_frames(video, n, W, H):
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    out = np.empty((n, H, W, 3), np.uint8)
    got = 0
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        out[i] = cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA),
                              cv2.COLOR_BGR2RGB)
        got += 1
    cap.release()
    return out[:got]


def smooth(uv, fps, cutoff=2.0):
    from scipy.signal import butter, filtfilt
    b, a = butter(2, cutoff / (0.5 * fps), btype="low")
    if len(uv) <= 12:
        return uv.copy()
    return filtfilt(b, a, uv.reshape(len(uv), -1), axis=0).reshape(uv.shape)


def background(frames, nsample=60):
    """Per-pixel temporal median. Valid because the camera is static."""
    k = max(1, len(frames) // nsample)
    return np.median(frames[::k], axis=0).astype(np.int16)


def foreground(frame, bg, thr=25):
    """Silhouette of the animal. Works even when it stands still, unlike
    frame-differencing, because the reference is the whole-clip median."""
    return np.abs(frame.astype(np.int16) - bg).mean(axis=2) > thr


def seed_points(fg, cx, cy, n=7, up=34, half=8, search=60):
    """Seed candidates ON the animal, anchored to the SILHOUETTE not the mesh.

    THE bug this exists to fix: the refined-sole pixel is the lowest vertex of
    the foot, so it lands on the floor just under the paw — measured 7-21 px
    from the nearest dog pixel. Seeding there tracks the floor, which is
    static and richly textured, so the tracker reported a perfectly stationary
    "paw" all clip (swing 0.07 px/frame against a true ~1.7).

    Seeding upward from the mesh point helped the front legs only, because it
    still inherits the mesh's vertical error. So the vertical origin is taken
    from the image instead: the lowest foreground row in a narrow column band
    around the paw IS the paw-floor contact, whatever the mesh thinks. Seeds
    go just above it, onto paw and pastern.
    """
    H, W = fg.shape
    x_lo, x_hi = int(max(0, cx - half)), int(min(W, cx + half + 1))
    y_lo, y_hi = int(max(0, cy - search)), int(min(H, cy + search // 2))
    if x_hi - x_lo < 2 or y_hi - y_lo < 4:
        return None
    band = fg[y_lo:y_hi, x_lo:x_hi]
    bots = []
    for c in range(band.shape[1]):
        rows = np.flatnonzero(band[:, c])
        if len(rows):
            bots.append(rows[-1] + y_lo)
    if len(bots) < 3:
        return None
    y_bot = float(np.median(bots))
    cand = []
    for dy in range(3, up, 2):                   # upward from the contact row
        for dx in range(-half, half + 1, 4):
            x, y = int(round(cx + dx)), int(round(y_bot - dy))
            if 4 <= x < W - 4 and 4 <= y < H - 4 and fg[y, x]:
                cand.append((x, y))
    if len(cand) < 2:
        return None
    cand = np.array(cand, float)
    d = np.linalg.norm(cand - np.array([cx, y_bot]), axis=1)
    return cand[np.argsort(d)[:n]]


def track_chunk(model, frames, x0, y0, seeds_xy, torch):
    """Track a seed cluster through one crop. Returns (T,2) full-frame + vis."""
    vid = torch.from_numpy(frames[:, y0:y0 + CH, x0:x0 + CW]).permute(
        0, 3, 1, 2)[None].float().cuda()
    q = np.stack([np.zeros(len(seeds_xy)), seeds_xy[:, 0] - x0,
                  seeds_xy[:, 1] - y0], axis=1).astype(np.float32)
    m = ((q[:, 1] > 4) & (q[:, 1] < CW - 4) & (q[:, 2] > 4) & (q[:, 2] < CH - 4))
    q = q[m]
    if len(q) < 2:
        return None, None
    with torch.no_grad():
        tr, vis = model(vid, queries=torch.from_numpy(q)[None].float().cuda(),
                        backward_tracking=False)
    tr = tr[0].cpu().numpy()
    vis = vis[0].cpu().numpy()
    keep = vis.mean(axis=0) > 0.6
    if keep.sum() < 2:
        keep = np.ones(len(q), bool)
    # median over the cluster: robust to a single point sliding onto floor
    return np.median(tr[:, keep], axis=1) + np.array([x0, y0]), vis[:, keep].mean(1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--infer", required=True, help="Phase B npz (seed + frame count)")
    p.add_argument("--contacts", default=None,
                   help="optional contacts npz; its refined sole pixels are "
                        "better seeds than the raw vertex-group centroid")
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=90)
    p.add_argument("--overlap", type=int, default=12)
    p.add_argument("--leash", type=float, default=20.0,
                   help="px of divergence from the mesh before re-seeding")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--fg-thr", type=float, default=25.0,
                   help="foreground threshold vs the temporal-median frame")
    p.add_argument("--seed-up", type=int, default=34,
                   help="px above the sole to search for on-animal seeds")
    p.add_argument("--cache", default="/media/SHARED_DATA/postcapitalistrobots/"
                   "animer/paw/cache")
    args = p.parse_args()

    os.environ.setdefault("TORCH_HOME", args.cache)
    import torch

    b = np.load(args.infer, allow_pickle=True)
    W, H = [int(v) for v in b["img_size"]]
    fps = float(b["fps"])
    N = int(b["num_frames"])
    if args.max_frames:
        N = min(N, args.max_frames)
    uv_mesh = (np.load(args.contacts, allow_pickle=True)["paw_uv"]
               if args.contacts else b["paw_uv"])[:N]
    uv_s = smooth(uv_mesh, fps)

    print(f"reading {N} frames at {W}x{H}", flush=True)
    frames = read_frames(args.video, N, W, H)
    N = len(frames)
    uv_mesh, uv_s = uv_mesh[:N], uv_s[:N]

    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline",
                           verbose=False).cuda().eval()
    print("computing background (temporal median)", flush=True)
    bg = background(frames)

    out = np.full((N, 4, 2), np.nan)
    visb = np.zeros((N, 4))
    reseeds = np.zeros(4, int)
    noseed = np.zeros(4, int)

    step = max(1, args.chunk - args.overlap)
    for leg in range(4):
        seed = None
        for s in range(0, N, step):
            e = min(N, s + args.chunk)
            if e - s < 4:
                break
            # crop follows the paw; if the paw traverses more than the crop
            # within a chunk the tracker would lose it at the border
            cx, cy = np.median(uv_s[s:e, leg], axis=0)
            x0 = int(np.clip(cx - CW // 2, 0, W - CW))
            y0 = int(np.clip(cy - CH // 2, 0, H - CH))
            if seed is None or not np.isfinite(seed).all():
                seed = uv_mesh[s, leg]
            elif np.linalg.norm(seed - uv_s[s, leg]) > args.leash:
                seed = uv_mesh[s, leg]          # leash: bounded, not jittery
                reseeds[leg] += 1
            # seed must lie inside the crop
            if not (x0 + 6 < seed[0] < x0 + CW - 6 and y0 + 6 < seed[1] < y0 + CH - 6):
                seed = uv_mesh[s, leg]
                x0 = int(np.clip(seed[0] - CW // 2, 0, W - CW))
                y0 = int(np.clip(seed[1] - CH // 2, 0, H - CH))
            fg = foreground(frames[s], bg, args.fg_thr)
            pts = seed_points(fg, seed[0], seed[1], up=args.seed_up)
            if pts is None:                     # paw not separable from bg here
                noseed[leg] += 1
                seed = None
                continue
            tr, vv = track_chunk(model, frames[s:e], x0, y0, pts, torch)
            if tr is None:
                seed = None
                continue
            # first chunk writes everything; later chunks skip the overlap,
            # which the previous chunk already covered with older evidence
            w0 = s if s == 0 else s + args.overlap // 2
            k = w0 - s
            out[w0:e, leg] = tr[k:]
            visb[w0:e, leg] = vv[k:]
            nxt = min(e, s + step)
            seed = tr[nxt - s - 1] if nxt - s - 1 < len(tr) else tr[-1]
        print(f"  {LEGS[leg]}: tracked, {reseeds[leg]} mesh re-seeds, "
              f"{noseed[leg]} chunks with no foreground seed", flush=True)

    bad = ~np.isfinite(out[..., 0])
    if bad.any():
        for leg in range(4):
            m = bad[:, leg]
            if m.any():
                idx = np.arange(N)
                good = ~m
                if good.sum() > 1:
                    for c in range(2):
                        out[m, leg, c] = np.interp(idx[m], idx[good], out[good, leg, c])
                else:
                    out[m, leg] = uv_mesh[m, leg]
    dev = np.linalg.norm(out - uv_mesh, axis=-1)
    print(f"\ntracker vs mesh pixel disagreement: median {np.median(dev):.1f} px, "
          f"p90 {np.percentile(dev, 90):.1f} px")
    print(f"mean visibility {visb.mean():.2f}; frames filled by interp "
          f"{100 * bad.mean():.1f}%")

    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o, source=str(b["source"]), fps=fps, num_frames=N,
                        paw_uv=out, visibility=visb, paw_uv_mesh=uv_mesh,
                        reseeds=reseeds, leash=args.leash,
                        tracker="cotracker3_offline")
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
