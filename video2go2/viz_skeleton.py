"""Render the Phase B/C output as 3D geometry, so it can be looked at.

There is no world placement yet -- Phase D has not run -- so the dog is drawn
root-centred and walks on the spot. What this does show is the thing that
feeds animal2go2: the ten canonical points, their articulation over time, and
which paws Phase C thinks are planted.

Writes an mp4 and a static strip of poses. The strip matters: a video is easy
to nod along to, and a row of frozen poses side by side makes a bad leg
obvious.
"""
from pathlib import Path
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection   # noqa: F401

LEGS = ["FR", "FL", "RR", "RL"]
# (parent, child) into the 10-point set:
# 0 root, 1 chest, 2-5 mounts FR FL RR RL, 6-9 toes FR FL RR RL
BONES = [(0, 1), (1, 2), (1, 3), (0, 4), (0, 5),
         (2, 6), (3, 7), (4, 8), (5, 9)]
LEG_COLOR = {"FR": "tab:red", "FL": "tab:orange",
             "RR": "tab:green", "RL": "tab:blue"}
TOE_OF = {6: "FR", 7: "FL", 8: "RR", 9: "RL"}


def to_display(p):
    """Camera frame (+x right, +y down, +z forward) -> Z-up display frame."""
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def draw(ax, pts, contact, trail=None, title=None, lim=0.55):
    ax.clear()
    for a, b in BONES:
        seg = pts[[a, b]]
        leg = TOE_OF.get(b)
        col = LEG_COLOR[leg] if leg else "0.25"
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c=col, lw=2.2)
    ax.scatter(*pts[:2].T, c="k", s=26)
    for i, leg in TOE_OF.items():
        down = contact[LEGS.index(leg)]
        ax.scatter(*pts[i], c=LEG_COLOR[leg], s=90 if down else 30,
                   edgecolors="k" if down else "none",
                   linewidths=1.4 if down else 0, depthshade=False)
    if trail is not None:
        for i, leg in TOE_OF.items():
            t = trail[:, i - 6]
            ax.plot(t[:, 0], t[:, 1], t[:, 2], c=LEG_COLOR[leg],
                    lw=0.9, alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=9)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--infer", required=True)
    p.add_argument("--contacts", required=True)
    p.add_argument("--out-video", default=None)
    p.add_argument("--out-strip", default=None)
    p.add_argument("--strip-range", default="174,330",
                   help="start,end frames for the pose strip")
    p.add_argument("--strip-n", type=int, default=5)
    p.add_argument("--metres-per-unit", type=float, default=1.133)
    p.add_argument("--stride", type=int, default=2, help="frame stride for video")
    p.add_argument("--body-frame", action="store_true",
                   help="strip global_orient, so the dog does not turn")
    args = p.parse_args()
    global VIEWS
    VIEWS = ([(8, -90, "side"), (89, -90, "top")] if args.body_frame
             else [(12, -72, "side-ish"), (78, -90, "top")])

    b = np.load(args.infer, allow_pickle=True)
    c = np.load(args.contacts, allow_pickle=True)
    fps = float(b["fps"])
    pts = b["points_local"] * args.metres_per_unit
    if args.body_frame:
        # global_orient^T lands in the SMAL body frame, which is ALREADY
        # +x forward, +y left, +z up. Applying the camera-frame permutation
        # on top of that puts "up" on the display Y axis and silently turns
        # the side view into a front view.
        pts = np.einsum("nji,nkj->nki", b["global_orient"][:, 0], pts)
    else:
        pts = to_display(pts)
    contacts = c["contacts"]
    valid = b["valid"]
    N = len(pts)

    # Re-centre vertically on the mean paw height so the dog sits in frame.
    pts = pts - pts[:, 6:10, 2].mean(axis=1)[:, None, None] * np.array([0, 0, 1.0])

    if args.out_strip:
        a, bb = [int(x) for x in args.strip_range.split(",")]
        idx = np.linspace(a, bb - 1, args.strip_n).astype(int)
        fig = plt.figure(figsize=(3.6 * args.strip_n, 8.2))
        for k, i in enumerate(idx):
            for row, (elev, azim, tag) in enumerate(
                    VIEWS):
                ax = fig.add_subplot(2, args.strip_n,
                                     row * args.strip_n + k + 1, projection="3d")
                ax.view_init(elev=elev, azim=azim)
                lo = max(0, i - 25)
                draw(ax, pts[i], contacts[i], trail=pts[lo:i + 1, 6:10],
                     title=(f"t={i/fps:.2f}s  {'' if valid[i] else '(interp)'}"
                            if row == 0 else tag))
        fig.suptitle("Phase B/C output — 10-point skeleton, root-centred. "
                     "Filled+outlined paw = Phase C says planted. "
                     "Faint lines are the last 25 frames of paw travel.",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(args.out_strip, dpi=95)
        print(f"wrote {args.out_strip}")
        plt.close(fig)

    if args.out_video:
        from matplotlib.animation import FFMpegWriter
        fig = plt.figure(figsize=(11, 5.6))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        writer = FFMpegWriter(fps=fps / args.stride, bitrate=3200)
        Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)
        with writer.saving(fig, args.out_video, dpi=100):
            for i in range(0, N, args.stride):
                lo = max(0, i - 25)
                ax1.view_init(elev=VIEWS[0][0], azim=VIEWS[0][1])
                draw(ax1, pts[i], contacts[i], trail=pts[lo:i + 1, 6:10],
                     title=f"t={i/fps:5.2f}s   "
                           f"{'detected' if valid[i] else 'INTERPOLATED'}   "
                           f"{contacts[i].sum()} feet down")
                ax2.view_init(elev=VIEWS[1][0], azim=VIEWS[1][1])
                draw(ax2, pts[i], contacts[i], trail=pts[lo:i + 1, 6:10],
                     title="top view")
                writer.grab_frame()
        print(f"wrote {args.out_video}")
        plt.close(fig)


if __name__ == "__main__":
    main()
