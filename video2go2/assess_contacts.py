"""Is Phase C good enough on the segment that matters -- the walk?

Three questions, none of which needs ground truth.

1. STANCE SLIP. A detected stance asserts "this paw is not moving". However
   far it actually moves during that run is error injected straight into the
   root trajectory by Phase D. This is the tolerance test; everything else is
   diagnosis.

2. DEPTH DEGENERACY. This dog walks toward the camera, so much of its paw
   travel is along the viewing direction, where motion produces almost no
   pixel displacement. If that is where the travel is, an image-space
   detector is structurally blind to it and no threshold fixes that.

3. AN INDEPENDENT CUE. Paw height relative to the other three paws, in the
   camera-oriented body frame, comes from theta and global_orient only -- no
   camera translation, no depth. The brief warns off "3D foot height", but
   that warning is about world height reconstructed through the camera. A
   *relative* height among the four paws has no camera in it at all. If it
   agrees with the speed cue, both are probably right.
"""
import sys
import numpy as np

sys.path.insert(0, "video2go2")
from contacts_2d import runs, lowpass

LEGS = ["FR", "FL", "RR", "RL"]
SEGMENTS = {
    "standing (far)": (0, 174),
    "WALKING": (174, 330),
    "milling (near)": (330, 660),
    "turn + away": (660, 807),
}


def main(infer_path, contact_path):
    b = np.load(infer_path, allow_pickle=True)
    c = np.load(contact_path, allow_pickle=True)
    fps = float(b["fps"])
    valid = b["valid"]
    paw_uv = b["paw_uv"]
    contacts = c["contacts"]
    focal = float(b["focal_full"])
    mpu = 1.133  # metres per SMAL unit

    pts_cam = b["points_local"] + b["root_model"][:, None, :] + b["cam_t"][:, None, :]
    depth = np.maximum(lowpass(pts_cam[:, 0, 2][:, None], fps, 1.0)[:, 0], 1e-3)
    m_per_px = mpu * depth / focal            # metres per pixel at the dog

    # ---- 1. stance slip -------------------------------------------------
    print("1. STANCE SLIP -- how far a 'planted' paw actually travels")
    print("   (metres; this is the error Phase D inherits per stance)\n")
    print(f"   {'segment':<18}{'n':>5}{'median':>9}{'p90':>9}{'max':>9}"
          f"{'med dur':>9}")
    for name, (a, bb) in SEGMENTS.items():
        slips, durs = [], []
        for leg in range(4):
            for s, e, v in runs(contacts[a:bb, leg]):
                if not v or e - s < 3:
                    continue
                s0, e0 = s + a, e + a
                if not valid[s0:e0].all():
                    continue
                track = paw_uv[s0:e0, leg]
                # peak-to-peak excursion, not endpoint drift: a paw that
                # wanders out and back is just as wrong.
                span_px = np.linalg.norm(track - track[0], axis=-1).max()
                slips.append(span_px * m_per_px[s0:e0].mean())
                durs.append((e - s) / fps)
        if not slips:
            print(f"   {name:<18}{0:>5}")
            continue
        sl = np.array(slips)
        print(f"   {name:<18}{len(sl):>5}{np.median(sl):>9.3f}"
              f"{np.percentile(sl, 90):>9.3f}{sl.max():>9.3f}"
              f"{np.median(durs):>9.2f}s")

    # ---- 2. depth degeneracy --------------------------------------------
    print("\n2. DEPTH DEGENERACY -- where does the paw travel actually go?")
    print("   fraction of 3D paw displacement along the camera axis, and how")
    print("   many pixels a 10 cm step produces in each direction\n")
    paws_cam = pts_cam[:, 6:10]
    d3 = np.diff(paws_cam, axis=0)
    for name, (a, bb) in SEGMENTS.items():
        seg = d3[a:min(bb, len(d3))]
        mag = np.linalg.norm(seg, axis=-1)
        big = mag > np.percentile(mag, 60)      # ignore near-stationary frames
        if big.sum() == 0:
            continue
        frac_z = np.abs(seg[..., 2])[big] / np.maximum(mag[big], 1e-9)
        z = depth[a:bb].mean()
        px_lat = focal * (0.10 / mpu) / z
        print(f"   {name:<18} along-camera-axis {frac_z.mean():.2f}   "
              f"10 cm sideways = {px_lat:5.1f} px,  10 cm in depth = "
              f"{focal*(0.10/mpu)*z/(z*z):5.1f} px at best")

    # ---- 3. independent height cue --------------------------------------
    print("\n3. INDEPENDENT CUE -- paw height relative to the other paws")
    print("   (camera-oriented body frame, +y is down; theta only)\n")
    paws_local = b["points_local"][:, 6:10]
    h = paws_local[..., 1]                      # +y down in this convention
    rel = h - np.median(h, axis=1, keepdims=True)
    rel = lowpass(rel, fps, 8.0) * mpu          # metres, + = lower than median
    lifted = -rel                               # + = raised above the pack

    for name, (a, bb) in SEGMENTS.items():
        m = valid[a:bb]
        if m.sum() == 0:
            continue
        seg_l = lifted[a:bb][m]
        seg_c = contacts[a:bb][m]
        if seg_c.sum() == 0 or (~seg_c).sum() == 0:
            print(f"   {name:<18} (no contrast)")
            continue
        print(f"   {name:<18} lifted when stance {seg_l[seg_c].mean()*100:+6.1f} cm"
              f"   when swing {seg_l[~seg_c].mean()*100:+6.1f} cm"
              f"   gap {(seg_l[~seg_c].mean()-seg_l[seg_c].mean())*100:5.1f} cm")

    # agreement between the two cues on the walking segment
    a, bb = SEGMENTS["WALKING"]
    m = valid[a:bb]
    hi_cue = (lifted[a:bb] < np.percentile(lifted[a:bb][m], 100 * (1 - 0.65)))
    agree = (hi_cue[m] == contacts[a:bb][m]).mean()
    print(f"\n   speed cue vs height cue agree on {100*agree:.1f}% of walking "
          f"leg-frames\n   (height cue thresholded to the same 0.65 duty)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
