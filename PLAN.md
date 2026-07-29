# PLAN — camera-height-generic placement

Goal: remove the pipeline's dependence on a favourably placed camera while
keeping the ground-plane assumption. Today the floor error is paw-height error
multiplied by **distance ÷ camera height** (STATUS.md, "The limiting factor"):
2.5× on dog_1, 4.9× on dog_2. The recommendation there — raise the camera —
fixes the clip, not the method. This plan fixes the method.

Everything new lives in `video2go2/`, as new files. The existing scripts stay
untouched and remain the baseline every phase is compared against.
`animal2go2` and the Milestone 1 npz contract are not modified.

---

## Root cause, restated precisely

`contacts_ground.py` and `world_place.py` both treat the homography output as
a **measurement**: paw pixel → ray → intersect z=0 → "the paw is there", per
frame, per paw. Ray∩plane asserts the along-ray position with full confidence,
when at a grazing angle that is exactly the direction the image barely
constrains. The error is strongly anisotropic:

- **across the viewing ray** — measured well at any camera height;
- **along the ray (depth)** — error × distance/height, unbounded as the view
  flattens.

The per-frame projections then contaminate two independent decisions: contact
*timing* (speed on the floor) and body *placement* (`solve_translation`, where
each planted paw proposes a full 3-vector). The fix is to stop projecting and
start estimating: floor positions become **variables in one clip-wide
optimization**, constrained by pixel-space residuals that automatically know
which directions the image does and does not pin down, with priors and an
independent depth cue filling the rest.

Honest bound: at extreme grazing angles the depth information genuinely is not
in the image. The plan bounds the damage — it does not conjure per-frame depth
from nothing.

---

## Phase 0 — synthetic harness with exact ground truth

**File:** `video2go2/synth_harness.py`
**Why first:** every later phase's claim is "error no longer scales with
d/h", and only the harness can measure that. STATUS.md already withdrew the
objection to it (AniMer recovers a rendered-mesh pose to 3.6% of body length).

- Render the SMAL mesh (same `SMALLayer`, frozen betas) walking a scripted
  trajectory — straight walk, a turn, a standing stretch — over a textured
  checkerboard floor, so `calibrate_ground.py` works on it unchanged.
- Sweep camera height × distance: at least {0.8, 1.1, 1.6, 2.2} m × {2, 4, 6} m.
  Same trajectory in every render.
- Emit ground truth: per-frame root pose, toe world positions, true contact
  flags, true focal, true camera pose, true metric scale.
- Run the **existing** pipeline end to end on every render and plot floor
  error, contact-timing F1, and recovered traverse length against d/h.

**Gate:** the baseline curve reproduces the linear scaling STATUS.md predicts.
If it does not, the error model is wrong and the rest of the plan is rethought
before any solver code is written.

---

## Phase 1 — contact timing that never touches the homography

**File:** `video2go2/contacts_kine.py`, same output contract as
`contacts_ground.py` (`contacts`, `valid`, `paw_uv`, plus `ground` carried
only as an initialization aid, clearly labelled as such).

Detect stance from signals that carry the 2–5 cm mesh noise but **not** the
d/h multiplier:

- **body-frame toe height** relative to the other toes, from the rigid FK
  already in `points_local` — a planted paw is at the bottom of the convex
  hull of the four;
- **vertical-velocity zero crossings** of the toe in the camera-oriented
  frame;
- **pixel-space paw speed** with hysteresis — with a static camera a planted
  paw is stationary in the image in both axes (`contacts_2d.py` already
  implements this; reuse, don't rewrite). Its known blind spot (motion along
  the viewing ray compresses to ~1 px per 10 cm, see the `contacts_ground.py`
  header) is why this is one vote among three, not the detector;
- fuse by voting, then the existing `refine_contacts` / `MIN_SEGMENT_S`
  hygiene.

Keep the bottom-of-foot pixel refinement (`refine_paw_pixels`) — that fix is
about *which pixel is the paw*, which Phase 2 needs regardless.

**Evaluation:** contact-timing F1 on the harness against true flags, at every
camera height; on the three real clips, compare against `contacts_ground.py`
with the per-clip tuned thresholds (dog_1 0.20, cat_1 0.35, dog_2 0.25).

**Gate:** F1 on the harness is flat across camera heights (that is the whole
point), and on real clips it is at least on par with the tuned baseline
*without* per-clip threshold tuning. Watch the known trap: duty factor is a
poor proxy — evaluate on timing of transitions, not aggregates.

---

## Phase 2 — joint placement (the core)

**File:** `video2go2/world_place_ba.py`. Reads the same inputs as
`world_place.py`, writes the same output keys, so `parse_video.py` and every
viz tool work unchanged.

### Variables

| block | size | notes |
|---|---|---|
| root translation spline | 3 × K knots | knots at ~0.15 s spacing (≈ the current 4 Hz low-pass), cubic |
| stance anchors | 2 × S | one (x, y) per stance run, z = 0 by construction |
| log metric scale `log s` | 1 | replaces `--metres-per-unit`; see Phase 3 |

Body orientation stays AniMer's, unfitted — the reasoning in
`world_place.py`'s docstring (translation-only is far better conditioned)
still holds; we are widening what counts as a measurement, not reopening the
rotation.

### Residuals

1. **Paw reprojection, in pixels.** Project the hypothesized world toe
   (`R_cw·(s·points_local[toe]) + T(t)` mapped through the calibrated K, R_cw,
   C from `camera_to_world`) and compare against the observed bottom-of-foot
   pixel track. This is the load-bearing change: at a grazing angle a large
   along-ray floor shift produces a tiny pixel residual, so the optimizer
   *knows* that direction is unconstrained and lets the priors carry it —
   instead of the homography asserting a wrong answer with zero stated
   uncertainty. The anisotropic weighting nobody has to hand-tune falls out
   here.
2. **Stance rigidity.** While planted (Phase 1 flags), world toe = its run's
   anchor, strong weight. Zero skate during stance becomes true by
   construction, not a post-hoc metric.
3. **FK consistency.** World toe minus root follows the frozen-shape skeleton,
   σ ≈ 2–5 cm (the measured paw-height noise).
4. **Smoothness.** Acceleration penalty on the root spline (subsumes the
   current low-pass), hinge penalty for toes below `z = floor_tol` (subsumes
   the post-hoc no-penetration lift).
5. **Apparent-size depth cue.** The frozen shape makes the animal its own
   depth ruler, and AniMer's weak-perspective `cam_t` already encodes it:
   under the assumed 5000 px focal, apparent size fixes
   `depth ≈ s · (focal_px / 5000) · cam_t_z`. Residual on the root's distance
   along the viewing ray, robustly weighted. Error grows with relative size
   error, **independent of camera height** — it constrains exactly the
   direction the grazing view cannot, which is what replaces raising the
   camera. (`focal_full` is still never used *geometrically*; here it is only
   the known convention `cam_t` was expressed in — the Traps rule survives.)

### Solver

`scipy.optimize.least_squares`, Huber loss on 1 and 5, explicit
`jac_sparsity` (each residual touches ≤ 2 spline knots + 1 anchor + scale —
the problem is a few thousand residuals over a few hundred variables, trivial
at these sizes). Initialize from the current pipeline: homography-projected
anchors (their across-ray component is good even when along-ray is not),
`solve_translation` + `fill_flight` for the spline, current fitted scale for
`s`. Convergence from that start should be fast; if it stalls, solve
scale-frozen first, then release `log s`.

### Evaluation

- **Harness:** floor error and traverse-length error vs d/h. **Gate:**
  sublinear — near-flat where the size cue dominates — where the baseline is
  linear.
- **Real clips**, via animal2go2's own unmodified retargeter, all four numbers
  at once per the "motionless robot" trap: clamp rate, skate, **and** traverse
  length, **and** the `viz_world.py` side-by-side looks right. Targets:
  dog_2 clamp 1.78% → below 1% and skate 0.031 → below 0.01 m/s **with its
  traverse preserved within 10%**; dog_1 and cat_1 must not regress.
- dog_1's weak tail (52% unanchored) is the stress test for the priors: the
  spline + FK + size cue should now degrade gracefully where
  `fill_flight`'s constant velocity guesses. If the tail becomes usable,
  drop the `--trim 0,660` note from STATUS.

---

## Phase 3 — scale inside the fit

Mostly falls out of Phase 2's `log s`, but it is a separate claim to verify:
today scale comes from paw spread on the calibrated floor
(`body_length_m` in `contacts_ground.py`), which **inherits the full d/h
amplification** — the fitted 0.580 m shoulder on dog_2 (4.9×) is suspect.
In the joint fit, scale is instead pinned by stance anchors + reprojection +
the size cue together.

**Gate:** on the harness, recovered scale error is flat across camera heights;
on the real clips, the two corridor clips (shared rig) should agree on the
camera height to ~1 cm *and* the dog_2 shoulder estimate should move toward
plausibility or stay put — either way, report it.

---

## Phase 4 — consolidation

- Wire `parse_video.py` diagnostics: per-frame marginal uncertainty of the
  root (from the Jacobian at the solution) goes into the `.video.json`
  sidecar — so a clip shot at a terrible angle *says so* instead of silently
  producing a confident wrong trajectory.
- Update STATUS.md: the "raise the camera" paragraph becomes "raising the
  camera still helps, but is no longer the limiting factor", with the new
  numbers.
- Retire per-clip contact thresholds if Phase 1's gate held.

---

## Later, explicitly out of scope now

- **Independent 2D paw detector** — slots in as a drop-in replacement for
  `paw_uv` in the reprojection residual, and becomes *more* valuable there:
  the formulation exists precisely to fuse error sources that are independent
  of the mesh fit. The measurement to justify it is already specified in
  STATUS.md.
- **Monocular metric depth** (e.g. a metric depth model) as a second
  camera-height-independent depth cue. Heavier dependency; only if the
  `cam_t` size cue proves too noisy (see risks).
- Migration into `animal2go2` — unchanged policy: not until trusted.

---

## Risks and checks

- **`cam_t` quality (Phase 2, residual 5).** AniMer predicts a per-crop
  weak-perspective camera; verify `animer_infer.py` stores the full-image
  translation (`pred_cam_t_full` or equivalent) and check its per-frame
  stability on the standing segments before trusting it as a depth cue. If it
  wobbles more than a few percent, down-weight it and lean on the anchors —
  the harness will show how much it was contributing.
- **Spline stiffness is a real knob.** Too stiff flattens the traverse (the
  motionless-robot trap, again); too loose readmits the 1–3 Hz drift. Tune
  once on the harness where truth is known, not per real clip.
- **Phase 1 fusion could inherit the mesh drift** — all three votes see the
  same mesh. They see it through different projections of the error, which is
  the defence, but the harness F1-vs-height curve is the check, not the
  argument.
- **Stale intermediates** faked a comparison once already (Traps). The
  harness runner clears its work directory per configuration.

## Order and effort

Phase 0 is the foundation and roughly a day of work; Phase 1 and Phase 2 are
independent of each other once 0 exists (contacts from Phase 1 feed Phase 2,
but Phase 2 can be developed against `contacts_ground.py` flags meanwhile).
Phase 2 is the bulk of the effort. Phases 3–4 are small. Each phase lands only
if its gate holds on both the harness and the three real clips.
