"""Ground-truth-free smoke test for ANY source of paw pixels.

Real video has no ground truth, so this checks NECESSARY conditions that any
correct paw measurement must satisfy. It cannot prove a source is right, but
it catches the failure that killed the CoTracker attempt — a "paw" track that
was actually frozen on floor texture, which looked superb on every stability
metric while being useless.

  1. STATIC WINDOWS. Windows where the raw pixels provably do not change
     (threshold calibrated against an empty background patch). A correct
     source reports ~0 motion. Large motion = hallucinated drift.
  2. MOTION WINDOWS. Windows where the pixels around the paw change a lot.
     A correct source MUST move. ~0 motion = locked to background. This is
     the check CoTracker failed (0.05-0.28 px/frame against a true ~1.7).
  3. AGREEMENT ENVELOPE vs the mesh. Not a correctness test — the mesh is
     what we are trying to beat — but sustained divergence of hundreds of px
     means the source is on a different object entirely.
  4. COVERAGE. Fraction of frames with a usable estimate.

A source that passes 1 AND 2 is measuring the paw. Passing 1 alone is the
trap: a floor-locked point passes 1 perfectly.
"""
from pathlib import Path
import argparse
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

LEGS = ["FR", "FL", "RR", "RL"]


def read_gray(video, n, W, H):
    import cv2
    cap = cv2.VideoCapture(video)
    g = np.empty((n, H, W), np.uint8)
    got = 0
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        g[i] = cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        got += 1
    cap.release()
    return g[:got]


def roi_diff_series(gray, uv_ref, R=24):
    """Per-frame mean |ΔI| in a box around each paw. ROI placement uses a
    SMOOTHED reference so its own jitter does not create fake motion."""
    n, H, W = gray.shape
    d = np.zeros((n, uv_ref.shape[1]))
    for p in range(uv_ref.shape[1]):
        for t in range(1, n):
            cx, cy = uv_ref[t, p]
            if not np.isfinite([cx, cy]).all():
                continue
            x0 = int(np.clip(cx - R, 0, W - 2 * R))
            y0 = int(np.clip(cy - R, 0, H - 2 * R))
            a = gray[t, y0:y0 + 2 * R, x0:x0 + 2 * R].astype(np.int16)
            b = gray[t - 1, y0:y0 + 2 * R, x0:x0 + 2 * R].astype(np.int16)
            d[t, p] = np.abs(a - b).mean()
    return d


def windows(mask, minlen):
    out, s = [], None
    for t in range(1, len(mask)):
        if mask[t] and s is None:
            s = t
        elif not mask[t] and s is not None:
            if t - s >= minlen:
                out.append((s, t))
            s = None
    if s is not None and len(mask) - s >= minlen:
        out.append((s, len(mask)))
    return out


def excursion(a):
    return np.linalg.norm(a - np.median(a, axis=0), axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--paw", required=True,
                   help="npz with paw_uv (N,4,2); paw_track / DLC / any source")
    p.add_argument("--mesh", required=True,
                   help="npz whose paw_uv is the mesh projection (baseline)")
    p.add_argument("--max-frames", type=int, default=660)
    p.add_argument("--empty", default="700,80",
                   help="y,x of a background patch with nothing moving, for "
                        "the sensor/compression noise floor")
    p.add_argument("--minlen", type=int, default=12)
    p.add_argument("--move-pct", type=float, default=88.0,
                   help="per-leg percentile of ROI change defining 'moving'")
    p.add_argument("--move-minlen", type=int, default=5)
    p.add_argument("--motion-stride", type=int, default=2,
                   help="frames between samples when measuring motion; >1 "
                        "bridges duplicated frames in converted video")
    p.add_argument("--drift-px", type=float, default=2.0,
                   help="static-window excursion above which a CORRELATED "
                        "error counts as hallucinated drift")
    p.add_argument("--jitter-px", type=float, default=8.0,
                   help="static-window excursion tolerated when the error is "
                        "INDEPENDENT (harness: 8px of white noise is as good "
                        "as perfect pixels)")
    p.add_argument("--move-smooth", type=int, default=5,
                   help="frames to smooth the ROI difference over; bridges "
                        "duplicated frames in frame-rate-converted video")
    args = p.parse_args()

    src = np.load(args.paw, allow_pickle=True)
    uv = src["paw_uv"]
    mesh = np.load(args.mesh, allow_pickle=True)["paw_uv"]
    n = min(len(uv), len(mesh), args.max_frames)
    uv, mesh = uv[:n], mesh[:n]

    from scipy.signal import butter, filtfilt
    b_, a_ = butter(2, 2.0 / (0.5 * 60), btype="low")
    ref = filtfilt(b_, a_, mesh.reshape(len(mesh), -1), axis=0).reshape(mesh.shape)

    H, W = 800, 1280
    gray = read_gray(args.video, n, W, H)
    n = len(gray)
    uv, mesh, ref = uv[:n], mesh[:n], ref[:n]

    ey, ex = [int(v) for v in args.empty.split(",")]
    emptyd = np.array([np.abs(gray[t, ey:ey + 48, ex:ex + 48].astype(np.int16) -
                              gray[t - 1, ey:ey + 48, ex:ex + 48].astype(np.int16)).mean()
                       for t in range(1, n)])
    floor = float(np.percentile(emptyd, 95))
    thr_static = max(floor * 1.6, 0.15)
    d = roi_diff_series(gray, ref)
    # dog_1.mov carries ~40% NEAR-DUPLICATE FRAMES (60 fps container, roughly
    # 36 fps of real content, irregularly duplicated). The raw ROI difference
    # therefore alternates high/low every frame and no run of consecutive
    # "moving" frames exists at all — the first version of this test found
    # zero motion windows and reported FAIL for lack of evidence, which would
    # have condemned a perfectly good source too. Smoothing over a few frames
    # bridges the duplicates and recovers genuine bursts of motion.
    k = args.move_smooth
    ker = np.ones(k) / k
    d_s = np.stack([np.convolve(d[:, i], ker, mode="same") for i in range(4)], 1)
    # Motion threshold is PER LEG and data-driven: each leg's own upper
    # quantile always yields the frames where that paw is most active.
    thr_move = np.percentile(d_s, args.move_pct, axis=0)
    d_move = d_s
    print(f"noise floor (empty patch p95) {floor:.2f} grey levels")
    print(f"  static below {thr_static:.2f} | moving above per-leg p"
          f"{args.move_pct:g} = " + ", ".join(f"{v:.2f}" for v in thr_move) + "\n")

    cov = np.isfinite(uv[..., 0]).mean()
    print(f"coverage: {100 * cov:.1f}% of paw-frames have an estimate\n")

    print(f"{'leg':>4} {'STATIC windows':>26} {'MOTION windows':>30}")
    print(f"{'':>4} {'n':>3} {'src px':>8} {'mesh px':>8} "
          f"{'n':>5} {'src px/fr':>10} {'mesh px/fr':>11}")
    verdict, ac = {}, {}
    for p_, leg in enumerate(LEGS):
        st = windows(d[:, p_] < thr_static, args.minlen)
        mv = windows(d_move[:, p_] > thr_move[p_], args.move_minlen)
        s_src = [np.median(excursion(uv[s:e, p_])) for s, e in st
                 if np.isfinite(uv[s:e, p_]).all()]
        s_msh = [np.median(excursion(mesh[s:e, p_])) for s, e in st]
        # Motion is measured over a STRIDE. dog_1 is frame-rate-converted and
        # ~23% of frames are duplicates; a per-frame detector returns an
        # identical answer on an identical frame, so per-frame displacement is
        # bimodal and its MEDIAN sits in the duplicate population — which made
        # this test read a perfectly good detector as "frozen" (0.38 px/frame,
        # against 1.8 px/frame over a 2-frame stride). The mesh escapes the
        # artefact only because AniMer low-passes pose at 6 Hz.
        k_ = args.motion_stride
        step_disp = lambda a: np.linalg.norm(a[k_:] - a[:-k_], axis=1) / k_
        m_src = [np.median(step_disp(uv[s:e, p_])) for s, e in mv
                 if np.isfinite(uv[s:e, p_]).all() and e - s > k_]
        m_msh = [np.median(step_disp(mesh[s:e, p_])) for s, e in mv
                 if e - s > k_]
        # Drift vs white jitter: lag-1 autocorrelation of the static-window
        # residual. The harness showed ~8 px of INDEPENDENT error is as
        # harmless as perfect pixels, while correlated error costs 2-3x — so
        # magnitude alone cannot judge a source. Mesh error is 1-3 Hz drift
        # (autocorr near 1); an honest per-frame detector is near 0.
        acs = []
        for s, e in st:
            r = uv[s:e, p_] - np.median(uv[s:e, p_], axis=0)
            if not np.isfinite(r).all() or len(r) < 6:
                continue
            r = r - r.mean(0)
            num = (r[1:] * r[:-1]).sum()
            den = (r * r).sum()
            if den > 1e-9:
                acs.append(num / den)
        ac[leg] = float(np.median(acs)) if acs else np.nan
        f = lambda v: f"{np.median(v):.2f}" if v else "  -  "
        print(f"{leg:>4} {len(st):>3} {f(s_src):>8} {f(s_msh):>8} "
              f"{len(mv):>5} {f(m_src):>10} {f(m_msh):>11}")
        verdict[leg] = (np.median(s_src) if s_src else np.nan,
                        np.median(m_src) if m_src else np.nan,
                        np.median(m_msh) if m_msh else np.nan)

    dev = np.linalg.norm(uv - mesh, axis=-1)
    print(f"\nagreement with mesh: median {np.nanmedian(dev):.1f} px, "
          f"p90 {np.nanpercentile(dev, 90):.1f} px")

    print("\nVERDICT (necessary conditions)")
    ok = True
    for leg, (s_, m_, mm_) in verdict.items():
        a_ = ac.get(leg, np.nan)
        drifty = (not np.isnan(a_)) and a_ > 0.5
        if np.isnan(s_):
            static_ok, why = True, "no static windows"
        elif drifty:
            static_ok = s_ < args.drift_px
            why = f"{s_:.2f}px CORRELATED (autocorr {a_:+.2f}) = drift"
        else:
            static_ok = s_ < args.jitter_px
            why = f"{s_:.2f}px independent (autocorr {a_:+.2f}) = jitter"
        move_ok = (not np.isnan(m_)) and (not np.isnan(mm_)) and m_ > 0.4 * mm_
        ok &= static_ok and move_ok
        print(f"  {leg}: static {'PASS' if static_ok else 'FAIL'} ({why})")
        print(f"      moves-with-paw {'PASS' if move_ok else 'FAIL'} "
              f"({m_:.2f} vs mesh {mm_:.2f} px/frame at stride {args.motion_stride})")
    print(f"\n{'PASS' if ok else 'FAIL'} — "
          + ("measuring the paw" if ok else
             "NOT a usable paw source (see failures above)"))


if __name__ == "__main__":
    main()
