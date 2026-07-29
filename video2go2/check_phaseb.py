"""Sanity-check a Phase B npz without any ground truth.

The visual paw overlay proves the projection is right; this proves the leg
*assignment* is right, which an overlay cannot when the dog faces the camera
and left/right are a mirror apart. Everything here is a sign or a ratio, so it
holds regardless of the unknown metric scale.
"""
import sys
import numpy as np

LEGS = ["FR", "FL", "RR", "RL"]


def main(path):
    d = np.load(path, allow_pickle=True)
    N = int(d["num_frames"])
    pts = d["points_local"]          # (N,10,3), root-centred, global_orient applied
    go = d["global_orient"][:, 0]    # (N,3,3) body->camera
    valid = d["valid"]
    fps = float(d["fps"])

    print(f"clip {str(d['source'])}: {N} frames @ {fps:.3f} fps, "
          f"{100*valid.mean():.1f}% detected")
    print(f"points_local {pts.shape}  paw_uv {d['paw_uv'].shape}")

    # Longest run of consecutive missed detections -- interpolating across a
    # long gap is fabrication, and Phase D needs to know where those are.
    gaps, run = [], 0
    for v in valid:
        run = 0 if v else run + 1
        gaps.append(run)
    print(f"longest detection gap: {max(gaps)} frames "
          f"({max(gaps)/fps*1000:.0f} ms)")

    # Undo global_orient to land in the canonical SMAL body frame, where the
    # template is +x forward, +y left, +z up.
    body = np.einsum("nji,nkj->nki", go, pts)   # R^T applied to each point

    mounts = body[:, 2:6]    # FR FL RR RL
    toes = body[:, 6:10]
    print("\nbody frame, medians over frames (SMAL units, +x fwd +y left +z up)")
    print(f"{'':<6}{'mount x':>9}{'mount y':>9}{'mount z':>9}"
          f"{'toe x':>9}{'toe y':>9}{'toe z':>9}")
    for i, leg in enumerate(LEGS):
        m = np.median(mounts[valid, i], axis=0)
        t = np.median(toes[valid, i], axis=0)
        print(f"{leg:<6}{m[0]:>9.3f}{m[1]:>9.3f}{m[2]:>9.3f}"
              f"{t[0]:>9.3f}{t[1]:>9.3f}{t[2]:>9.3f}")

    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'ok ' if cond else 'FAIL'}] {name}")

    print("\nassignment checks")
    my = np.median(mounts[valid], axis=0)[:, 1]
    ty = np.median(toes[valid], axis=0)[:, 1]
    check("FL/RL mounts are +y (left)", my[1] > 0 and my[3] > 0)
    check("FR/RR mounts are -y (right)", my[0] < 0 and my[2] < 0)
    check("FL/RL toes are +y (left)", ty[1] > 0 and ty[3] > 0)
    check("FR/RR toes are -y (right)", ty[0] < 0 and ty[2] < 0)
    mx = np.median(mounts[valid], axis=0)[:, 0]
    check("front mounts ahead of rear", min(mx[0], mx[1]) > max(mx[2], mx[3]))
    tz = np.median(toes[valid], axis=0)[:, 2]
    mz = np.median(mounts[valid], axis=0)[:, 2]
    check("toes below their mounts", (tz < mz).all())

    # Scale: what would this dog measure if the SMAL units were metres?
    trunk = np.linalg.norm(body[valid, 1] - body[valid, 0], axis=-1)
    shoulder_h = np.median(mounts[valid, :2, 2] - toes[valid, :2, 2])
    print(f"\nscale reference (SMAL units)")
    print(f"  pelvis->chest        {np.median(trunk):.4f}")
    print(f"  front leg length     {shoulder_h:.4f}")
    print(f"  a golden retriever's shoulder is ~0.58 m, so 1 SMAL unit "
          f"~= {0.58/shoulder_h:.3f} m")
    print(f"  -> implied trunk {np.median(trunk)*0.58/shoulder_h:.3f} m "
          f"(plausible range 0.45-0.75 m for the breed)")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
