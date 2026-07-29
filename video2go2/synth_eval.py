"""Phase 0 (PLAN.md) — run a placement pipeline over the synthetic configs
and measure it against exact ground truth.

For each cfg_* directory made by synth_harness.py:
  1. clear the per-config work directory (Traps: stale intermediates fake
     comparisons);
  2. run contacts_ground.py with ONE fixed policy across all configs (no
     per-config tuning — the point is genericity);
  3. fit the scale the way the real pipeline does, from paw spread on the
     calibrated floor vs the same spread in model units;
  4. run the placement stage (world_place.py, or world_place_ba.py with
     --solver ba once Phase 2 exists);
  5. compare to truth: root error, stance-toe floor error, traverse length,
     contact F1, scale error.

Aggregates into <outdir>/baseline_<tag>.json and a curve of error vs the
amplification factor distance/height. STATUS.md predicts the baseline is
LINEAR in it; Phase 2 must break that.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from contacts_2d import runs

PY = "/home/alejodosr/anaconda3/envs/animal/bin/python"
V2G = Path(__file__).parent
TOE0 = 6


def run(cmd, log_path):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                       cwd=str(V2G.parent))
    Path(log_path).write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-15:])
        raise RuntimeError(f"{cmd[1]} failed (log {log_path}):\n{tail}")
    return r.stdout


def fit_scale(animer_npz, calib_json):
    """The baseline scale fit: paw spread on the floor / paw spread in units.

    Mirrors STATUS.md ('one scalar per clip, from the animal's own paw
    spacing on the calibrated floor') — and therefore inherits the floor
    error, which is exactly what Phase 3 wants measured.
    """
    from contacts_ground import body_length_m
    b = np.load(animer_npz, allow_pickle=True)
    cal = json.loads(Path(calib_json).read_text())
    body_m = body_length_m(np.array(cal["H"]), b["paw_uv"])
    paws = b["points_local"][:, TOE0:TOE0 + 4]
    spread_u = np.median(np.linalg.norm(paws.max(axis=1) - paws.min(axis=1),
                                        axis=-1))
    return float(body_m / max(spread_u, 1e-9))


def contact_f1(est, gt):
    tp = float((est & gt).sum())
    p = tp / max(float(est.sum()), 1.0)
    r = tp / max(float(gt.sum()), 1.0)
    return 2 * p * r / max(p + r, 1e-9), p, r


def evaluate(world_npz, gt_motion, gt_cfg):
    w = np.load(world_npz, allow_pickle=True)
    world_est, contacts_est = w["world"], w["contacts"]
    world_gt, contacts_gt = gt_motion["world"], gt_motion["contacts"]
    n = min(len(world_est), len(world_gt))
    world_est, world_gt = world_est[:n], world_gt[:n]
    contacts_est, contacts_gt = contacts_est[:n], contacts_gt[:n]

    root_err = np.linalg.norm(world_est[:, 0, :2] - world_gt[:, 0, :2], axis=1)
    z_err = np.abs(world_est[:, 0, 2] - world_gt[:, 0, 2])

    # stance-toe floor error, on GT stance frames: the quantity the homography
    # amplifies. Uses GT stance so every pipeline is scored on the same frames.
    te = []
    for leg in range(4):
        m = contacts_gt[:, leg]
        if m.any():
            te.append(np.linalg.norm(world_est[m, TOE0 + leg, :2]
                                     - world_gt[m, TOE0 + leg, :2], axis=1))
    toe_err = np.concatenate(te) if te else np.array([np.nan])

    def pathlen(x):
        return float(np.linalg.norm(np.diff(x[:, :2], axis=0), axis=1).sum())

    f1, prec, rec = contact_f1(contacts_est, contacts_gt)
    return {
        "root_err_med_m": float(np.median(root_err)),
        "root_err_p90_m": float(np.percentile(root_err, 90)),
        "root_z_err_med_m": float(np.median(z_err)),
        "toe_floor_err_med_m": float(np.median(toe_err)),
        "toe_floor_err_p90_m": float(np.percentile(toe_err, 90)),
        "traverse_ratio": pathlen(world_est[:, 0]) / max(pathlen(world_gt[:, 0]), 1e-9),
        "contact_f1": f1, "contact_precision": prec, "contact_recall": rec,
        "amplification": float(gt_cfg["amplification"]),
        "height": float(gt_cfg["height"]), "distance": float(gt_cfg["distance"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synth", default="/media/SHARED_DATA/postcapitalistrobots/"
                   "animer/v2go2/synth")
    p.add_argument("--tag", default="baseline")
    p.add_argument("--solver", choices=["baseline", "ba"], default="baseline",
                   help="baseline = contacts_ground + world_place; "
                        "ba = contacts_kine + world_place_ba (Phases 1-2)")
    p.add_argument("--thresh", type=float, default=0.20,
                   help="contacts_ground threshold, body-lengths/s; ONE value "
                        "for every config, matching dog_1's tuned baseline")
    p.add_argument("--configs", default="", help="comma list to restrict")
    p.add_argument("--true-scale", action="store_true",
                   help="feed the GT metres-per-unit instead of fitting it: "
                        "isolates placement error from scale error")
    p.add_argument("--plot", action="store_true", default=True)
    args = p.parse_args()

    synth = Path(args.synth)
    gt_motion = np.load(synth / "gt_motion.npz", allow_pickle=True)
    cfgs = sorted(d for d in synth.iterdir()
                  if d.is_dir() and d.name.startswith("cfg_"))
    if args.configs:
        keep = set(args.configs.split(","))
        cfgs = [c for c in cfgs if c.name in keep]

    results = {}
    print(f"{'config':>14} {'amp':>5} {'scale err':>9} {'F1':>5} "
          f"{'toe med':>8} {'root med':>9} {'traverse':>8}")
    for cfg in cfgs:
        work = cfg / f"work_{args.tag}"
        if work.exists():
            shutil.rmtree(work)          # no stale intermediates, ever
        work.mkdir()
        animer, calib = cfg / "animer.npz", cfg / "calib.json"
        gt_cfg = np.load(cfg / "gt.npz", allow_pickle=True)
        try:
            if args.solver == "baseline":
                run([PY, V2G / "contacts_ground.py", "--infer", animer,
                     "--calib", calib, "--out", work / "contacts.npz",
                     "--thresh", args.thresh, "--segments", "0,174,330,660"],
                    work / "contacts.log")
                mpu_est = (float(gt_motion["metres_per_unit"])
                           if args.true_scale else fit_scale(animer, calib))
                run([PY, V2G / "world_place.py", "--infer", animer,
                     "--contacts", work / "contacts.npz", "--calib", calib,
                     "--out", work / "world.npz",
                     "--metres-per-unit", mpu_est], work / "world.log")
            else:
                run([PY, V2G / "contacts_kine.py", "--infer", animer,
                     "--calib", calib, "--out", work / "contacts.npz"],
                    work / "contacts.log")
                run([PY, V2G / "world_place_ba.py", "--infer", animer,
                     "--contacts", work / "contacts.npz", "--calib", calib,
                     "--out", work / "world.npz"], work / "world.log")
                w = np.load(work / "world.npz", allow_pickle=True)
                mpu_est = float(w["metres_per_unit"])
        except RuntimeError as e:
            print(f"{cfg.name:>14}  FAILED: {str(e).splitlines()[0]}")
            results[cfg.name] = {"failed": str(e)[:2000]}
            continue

        m = evaluate(work / "world.npz", gt_motion, gt_cfg)
        m["scale_est"] = mpu_est
        m["scale_err_frac"] = mpu_est / float(gt_motion["metres_per_unit"]) - 1.0
        results[cfg.name] = m
        print(f"{cfg.name:>14} {m['amplification']:5.2f} "
              f"{m['scale_err_frac']:+9.1%} {m['contact_f1']:5.2f} "
              f"{m['toe_floor_err_med_m']:8.3f} {m['root_err_med_m']:9.3f} "
              f"{m['traverse_ratio']:8.2f}")

    out_json = synth / f"{args.tag}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_json}")

    good = {k: v for k, v in results.items() if "failed" not in v}
    if len(good) >= 3:
        amp = np.array([v["amplification"] for v in good.values()])
        toe = np.array([v["toe_floor_err_med_m"] for v in good.values()])
        slope = np.polyfit(amp, toe, 1)
        r = np.corrcoef(amp, toe)[0, 1]
        print(f"toe floor error vs amplification: slope {slope[0] * 100:.1f} "
              f"cm per unit d/h, intercept {slope[1] * 100:.1f} cm, "
              f"corr {r:.2f}")
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
            for ax, key, nm in zip(
                    axes, ["toe_floor_err_med_m", "root_err_med_m",
                           "traverse_ratio"],
                    ["stance toe floor error (m)", "root error (m)",
                     "traverse ratio (1 = truth)"]):
                for k, v in good.items():
                    ax.scatter(v["amplification"], v[key], s=42)
                    ax.annotate(f"h{v['height']:g}/d{v['distance']:g}",
                                (v["amplification"], v[key]), fontsize=7,
                                xytext=(4, 3), textcoords="offset points")
                ax.set_xlabel("amplification  distance / height")
                ax.set_title(nm)
                ax.grid(alpha=.3)
            axes[2].axhline(1.0, color="k", lw=.8, ls="--")
            fig.suptitle(f"{args.tag}: error vs camera geometry")
            fig.tight_layout()
            fig.savefig(synth / f"{args.tag}_curve.png", dpi=110)
            print(f"wrote {synth / (args.tag + '_curve.png')}")


if __name__ == "__main__":
    main()
