# STATUS — video → Go2

Turning a monocular video of an animal into the npz that `animal2go2` already
retargets onto a Unitree Go2. Working end to end on three clips.

**animal2go2 is never modified.** Everything here lives in `video2go2/` and
produces a file that its unmodified Milestone 1 pipeline consumes.

---

## DEFAULT PIPELINE (2026-07-31) — no manual ground calibration

`video2go2/run_default.sh <video> <clip> [trim]`

    AniMer pose -> DLC 2D skeleton -> [pose refinement] -> ZoeDepth ground
    plane -> contacts_kine -> world_place_ba (--size-prior 0) -> parse_video

The clicked four-point calibration is NO LONGER used. The ground plane comes
from ZoeDepth metric depth, which removes both the interactive step and the
tile-size assumption that made the old metres provisional. Scale carries no
biological prior (ablation: 6-18% of the weight, <=4.6% effect, all sigmas
hand-asserted).

Default outputs under `v2go2/default/`, side-by-sides
`paw/work/videos/DEFAULT_sbs_*.mp4`:

| clip | scale | shoulder | traverse | clamp | skate after |
|---|---|---|---|---|---|
| dog_1 | 1.091 | 54.2 cm | 4.74 m | 1.45% | 0.020 |
| dog_2 | 0.753 | 40.6 cm | 4.22 m | 5.56% | 0.097 |
| cat_1 | 0.556 | 38.9 cm | 2.33 m | 7.35% | 0.062 |

Known limitation carried deliberately: on dog_2 (Veo-generated) the depth
plane implies a 40.6 cm dog, 65% below the clicked calibration. Depth models
have no real geometry to measure in generated video. A one-sided
biological-plausibility REJECTION test on the recovered plane is the intended
guard and is not yet implemented.

The older clicked-calibration path still exists in `calib/` and is what the
`baseline` rows elsewhere in this file refer to.

## The pipeline

```
video ──1── AniMer pose ──2── ground calibration ──3── contacts
                                     │                    │
                                     └────────4── world placement
                                                          │
                                              5── npz (Milestone 1 contract)
                                                          │
                                      animal2go2, unmodified: retarget → Go2
```

### 1. `animer_infer.py` — pose per frame

Runs AniMer under the `animal` conda env. Per frame it keeps SMAL pose, shape
and the camera term, then:

- **freezes shape to the per-clip median** — AniMer re-estimates shape every
  frame and it wobbles 2–6%, which would make the skeleton change size;
- **smooths pose in 6D rotation representation**, not axis-angle (wraparound);
- **re-runs FK** and extracts ten canonical points: pelvis, chest, four leg
  roots, four paws, in `FR, FL, RR, RL` order.

Output: `<clip>_animer.npz` — points in a camera-oriented, root-centred frame,
plus paw pixel tracks.

### 2. `calibrate_ground.py` — the floor

Click four points on the floor that form a rectangle, say how many metres
across and along. Gives a homography mapping any floor pixel to a floor
position. **The focal length is solved from the rectangle**, so nothing needs
measuring — and the residual non-perpendicularity of the recovered ground axes
is reported as a check that the clicked quad really is the shape claimed.

One JSON per camera setup. Reused across clips from that setup.

### 3. `contacts_ground.py` — which paws are planted

Paw pixel → homography → position on the floor → is it moving? Below a
threshold, the paw is planted.

The paw point is the **bottom of the foot**, taken as the lowest vertex of the
foot region along the world up direction — not the centre of the SMAL vertex
group, which sits 1.8–3.0 cm higher (see Traps).

### 4. `world_place.py` — where the animal is

The camera is static and AniMer already gives body orientation every frame, so
the only per-frame unknown is a translation — far better conditioned than a
rigid fit over three noisy points. Camera→world comes from the homography by
Zhang's construction. Each planted paw proposes a position; with three or more
down, the worst-fitting one is rejected rather than averaged in. Frames with
nothing planted are interpolated, with a no-penetration constraint so feet
cannot sink through the floor.

### 5. `parse_video.py` — the handoff

Emits the nine-key Milestone 1 contract (metres, Z-up, ground at z≈0,
`FR FL RR RL`, xyzw quaternions) and validates it: shapes, unit-norm quats,
trunk length in a plausible range, root above toes, leg order, front-ahead-of-
rear. Diagnostics go to a sidecar `.video.json`, never into the npz.

**Scale is fitted, not assumed.** One scalar per clip, from the animal's own
paw spacing on the calibrated floor. It does not need to be told what species
it is looking at.

### Visual checks

| tool | shows |
|---|---|
| `viz_mesh_overlay.py` | mesh + floor grid + paws + **paw drift trails** on the source |
| `viz_world.py` | source beside the 3D skeleton, or beside the Go2 |
| `check_phaseb.py` | leg assignment, independent of the left/right mirror |
| `assess_contacts.py` | stance slip, the tolerance test |

---

## Running it

```bash
P=/home/alejodosr/anaconda3/envs/animal/bin/python
V=/media/SHARED_DATA/postcapitalistrobots/animer/v2go2

# 1 — pose  (needs the 8.35 GB vith checkpoint)
PYOPENGL_PLATFORM=egl FVCORE_CACHE=.../detectron2_cache \
$P video2go2/animer_infer.py --video media/dog_2.mp4 --out $V/dog_2_animer.npz

# 2 — floor  (interactive: click 4 tile corners, enter metres)
$P video2go2/calibrate_ground.py --video media/dog_2.mp4 \
   --camera dog_2_room --out calib/dog_2_room.json

# 3,4,5
$P video2go2/contacts_ground.py --infer $V/dog_2_animer.npz \
   --calib calib/dog_2_room.json --out $V/dog_2_contacts_ground.npz --thresh 0.25
$P video2go2/world_place.py --infer $V/dog_2_animer.npz \
   --contacts $V/dog_2_contacts_ground.npz --calib calib/dog_2_room.json \
   --out $V/dog_2_world.npz --metres-per-unit <fitted>
$P video2go2/parse_video.py --world $V/dog_2_world.npz --out $V/processed/dog_2.npz

# then, in animal2go2, unmodified
.venv/bin/python retarget/retarget.py dog_2.npz
MUJOCO_GL=egl .venv/bin/python viz/playback.py motions/dog_2.pkl
```

Omitting `--thresh` solves it from a target duty factor. That self-calibrates
but is consistently too permissive; per-clip values below are better.

---

## Where it stands

Reported by animal2go2's own retargeter. Clamp rate warns above 3%.

| clip | source | clamp | skate after | anchored | fitted size |
|---|---|---|---|---|---|
| `dog_1` | real, 60 fps, 13.4 s | 0.17% | 0.001 m/s | 71% | 0.555 m shoulder |
| `cat_1` | real, 60 fps, 6.6 s | 0.08% | 0.003 m/s | 76% | 0.305 m shoulder |
| `dog_2` | Veo, 24 fps, 8.0 s | 1.78% | 0.031 m/s | 66% | 0.580 m shoulder |

Per-clip thresholds: dog_1 `0.20`, cat_1 `0.35`, dog_2 `0.25` (body-lengths/s).

Calibrations, each solved independently: focal 1012 / 1056 / 1099 px, camera
height 1.10 / 1.09 / 1.14 m. The two corridor clips share a camera rig and
their heights agree to 1 cm, which is the strongest evidence the method works.

---

## The limiting factor

Reconstructed paw height is good to **2–5 cm**. The homography turns that into
ground error multiplied by **distance ÷ camera height** — 2.5× on dog_1, 4.9×
on dog_2. So a couple of centimetres of pose error becomes 5–20 cm of apparent
sliding, which is what causes missed contacts and a compressed trajectory.
Those are one defect, not two.

Measured on frames where the animal is provably standing still, the error is
smooth 1–3 Hz drift, **not** jitter — the same frequency band as real paw
motion, so no filter separates them, and more resolution does not help (2× the
pixels changed it by 1%).

**Biggest available win is not in this code: raise the camera and keep the
animal closer.** Camera at 2 m with the animal at 3 m gives ~1.5× instead of
4.9× — better than three times less error, for free.

*Update (see "Camera-generic path" below): the synthetic harness confirmed
this error model exactly — baseline floor error is 9.3 cm per unit of d/h,
correlation 0.98 — and the joint-placement path cuts the slope to 1.1 cm per
unit, with the d/h dependence shown to enter almost entirely through the
SCALE fit, not the placement. On the harness, raising the camera is no longer
the limiting factor. On real clips this is so far only proven for the
across-view clip (dog_1); the along-view clips still lose to the baseline at
the retargeter (details below).*

After that, an independent 2D paw detector. Not because it would be more
precise per frame, but because its errors would be *independent* of the
whole-body mesh fit, which is what we currently lack.

---

## Traps

Things that cost time, so they do not cost it twice.

**AniMer.** `pred_keypoints_3d` is 26 surface landmarks, not the skeleton —
call `SMALLayer.forward` unbound to get the rigid joints, because
`J_regressor @ posed_vertices` leaks ~1% length variation into rigid bones.
The four paws are vertex groups `[4, 3, 6, 5]` for FR/FL/RR/RL — the front
pair is transposed relative to reading order. The working checkpoint is the
8.35 GB `gdrive_AniMer` one; the 2.7 GB one declares an unsupported backbone.

**`cv2.CAP_PROP_FPS` lies.** On dog_1 it reports 58.861 where every real
inter-frame delta is 60.000. Use ffprobe. A 1.9% error biases every velocity.

**Never pass AniMer's `focal_full` to anything geometric.** It is an assumed
5000 px — a 14.6° lens. Using it put the camera below the floor.

**Four points always fit a homography exactly**, so a 0.00 px residual proves
nothing. Three near-collinear points give a perfect fit that is wrong
everywhere else; `calibrate_ground.py` rejects that configuration explicitly.

**Clamp rate and skate are both minimised by a motionless robot.** A "perfect"
run — 0.02% clamp, 0.001 m/s skate — turned out to have collapsed the cat's
traverse from 1.43 m to 0.30 m. Never optimise either without also checking
the trajectory survives.

**Duty factor is a poor proxy for contact quality.** On dog_1, moving the
threshold from 0.20 to 0.30 changes duty by 0.01 and quintuples the clamp
rate: a few individually bad contacts, invisible to any aggregate.

**Stale intermediates fake a comparison.** A failed stage left an old file that
the next stage consumed; two rows of a sweep came out byte-identical. Clear
intermediates between runs.

### Tried and rejected, with measurements

- **Windowed-excursion contact metric** instead of speed. Sound reasoning (the
  error and the signal share a frequency band, so discriminate on amplitude).
  Measured: clamp 0.34% → 2.21% (dog), 4.42% (cat). To reach a realistic duty
  it must admit ~7 cm of wander.
- **Body-length-normalised thresholds** to unify clips. Kept, because the units
  are interpretable — but it does **not** unify them. The dog stands still most
  of its clip and the cat walks most of its, so the same threshold lands at
  different points on each speed distribution.
- **Relabelling contacts from 3D height** after placement. Raises anchoring
  from 66% to 87%, but clamp 1.8% → 4.3% at every tolerance from 0.02 to 0.06.
  Paw height is only good to 2–5 cm, wider than the decision boundary.
- **Doubling inference resolution.** 1% change in paw jitter.

---

## Camera-generic path (PLAN.md, 2026-07-29)

New files, existing pipeline untouched: `synth_harness.py` + `synth_eval.py`
(Phase 0), `contacts_kine.py` (Phase 1), `world_place_ba.py` (Phase 2/3).

**Harness** (dog_1's recovered motion re-filmed from 12 synthetic cameras,
heights 0.8–2.2 m × distances 2–6 m, exact ground truth, shared noise
realization of 3.7 cm median paw error):

| metric, across d/h 0.9–7.5 | baseline | joint (BA) |
|---|---|---|
| stance-toe floor error slope | 9.3 cm per unit d/h (corr .98) | **1.1 cm per unit** |
| worst-geometry toe error | 0.66 m | **0.13 m** |
| fitted scale error | −4% … **+184%** | −12% … +1% |
| traverse ratio | 0.8 … **5.8×** | 0.90 … 0.99 |
| contact F1 (vs declared GT) | 0.25–0.32, geometry-dependent | **0.61–0.62, flat** |

The single most important measurement: with TRUE scale supplied, the
baseline's floor error is nearly flat in d/h — the amplification enters
almost entirely through the paw-spread scale fit. Fixing scale fixes most of
the camera-height sensitivity.

**Contacts** (`contacts_kine.py`): votes over forward-relative toe velocity
(the strong signal — stance is backward relative motion at ~walking speed,
against 2–5 cm noise), body-frame toe height, and pixel speed. ONE set of
constants lands biomechanically plausible duty on all three real clips
(dog_1 walking 0.67, cat_1 0.66, dog_2 0.63) where the floor-speed detector
needed per-clip thresholds and left 41% of walking frames with zero feet.

**Real clips through the unmodified retargeter** (clamp / skate m/s after):

| clip | baseline | BA path |
|---|---|---|
| dog_1 (across-view) | 0.06% / 0.000 | 0.61% / 0.006 ✓ |
| cat_1 (along-view) | 0.08% / 0.003 | **10.4% / 0.081 ✗** |
| dog_2 (along-view) | 1.78% / 0.031 | **7.3% / 0.102 ✗** |

The along-view regression was debugged to three causes; two are fixed, one
is structural (numbers after fixes: cat 10.4→7.2%, dog_2 ~8%, dog_1 0.6%):
1. **Over-merged stance runs + unconditional pinning** — the kinematic
   detector merges footfalls on slow gaits, so one anchor stood for real toe
   travel and pinning stretched emitted legs to 1.05× their own FK length
   (p99). Fixed: runs split at 0.25 leg-lengths of floor travel
   (`--split-budget`), and runs whose pin correction exceeds 0.12 leg-lengths
   are left on FK (`--max-pin-frac`).
2. **Body floating off its planted feet** — the stiff hinge, depth cue and
   smoothness out-voted the stance tie (stance-toe z p90 was 6.8 cm). Fixed:
   stance-z tie 4 cm → 1.5 cm (`--sigma-fkz`), hinge softened to 3 cm.
3. **Structural, unfixed**: the BA motion is SMOOTH, so animal2go2's initial
   contact gate (z < 3 cm AND speed < 0.25 m/s) accepts ~90% of frames as
   stance — arguably correct for a milling cat — and `pin_stance_feet` then
   holds each foot at one spot for seconds, integrating our residual
   trunk-vs-anchor drift (~0.05 m/s) into leg stretch. The baseline evades
   this only because its jittery feet FAIL the speed gate, keeping pin spans
   short. The retargeter's constants are implicitly co-tuned with the
   baseline's noise; fixing it properly means revisiting them at migration
   time (animal2go2 is read-only until then), or eliminating the residual
   drift, which needs the independent paw detector. Injecting jitter to game
   the gate was considered and rejected (the motionless-robot trap, inverted).

Until then, **the baseline remains the production path for along-view
clips**; the BA path wins wherever truth is measurable (harness) and on
across-view footage. (`--falsify-z` exists to drop solver-contradicted
stance samples; measured harmful — freed toes stay slow-and-low and get
pinned anyway, now unanchored — so it defaults off.)

**Scale, three estimators** (`world_place_ba.py` prints all three):
- *lateral-anchor* — pair separations of ray-only stance anchors projected on
  the image-lateral direction, where the homography is amplification-free.
  Unbiased to −12…+1% on the harness at every camera. Needs lateral
  information: an along-view walk leaves only the ~0.1-unit left-right paw
  spacing and it degenerates (info 114 on dog_1 vs 19–33 on dog_2/cat_1).
- *floor-spread* (the baseline fit) — inherits the full d/h amplification:
  +185% on dog_2's 4.9× geometry, exactly as the harness predicted.
- *depth cue* (AniMer's `cam_t` apparent size) — **precise but biased**:
  1% relative std over dog_1's standing segment, +35% absolute (arbitrated
  against the standing-segment floor truth of 0.877 m/unit). A stability
  check alone would have blessed it; it is used only as RELATIVE depth, with
  a per-clip bias factor estimated under a weak prior.
  Policy: lateral-anchor when its information is sufficient, else the median
  of the three. The honest error bar on along-view absolute scale is ±20%
  until the independent 2D paw detector exists.

### New traps

**`cam_t` pairs with the RAW FK frame, not the root-centred one.** The raw
SMAL root sits ~0.3 units from the origin and raw FK transforms as
R@(x−j0)+j0. Fabricating observations with the wrong convention shifted the
whole world by a constant 0.31 m — laterally, so it looked like a mystery
bias, not a frame error.

**The retargeter names its output from the npz `source` field, not the
filename.** Feeding it an experiment npz with `source=dog_2` silently
overwrites `motions/dog_2.pkl`. Pass `--source <name>_ba` to parse_video for
any experimental branch. (All three baseline pkls were regenerated from the
baseline processed npz after learning this.)

**Scale must not be a least-squares variable.** Every residual it multiplies
also multiplies the FK noise, so the optimizer shrinks the animal to shrink
the noise: a stable −8…−25% (errors-in-variables attenuation, worsened by the
Huber knee). Estimate scale outside the LS from time-averaged geometry.

## Paw-measurement investigation (2026-07-31)

Two results, and the second overturns the premise the first was chasing.

### A point tracker cannot track these paws — rejected, with mechanism

CoTracker3 (SOTA point tracking) on native-resolution 512x384 crops — the
crop matters: its predictor resizes to 384x512, so feeding it the 1280x800
frame discards a factor of 2.5 and would have faked a null result.

In windows where the RAW PIXELS provably do not change (threshold calibrated
on an empty floor patch, noise floor p95 = 0.13 grey levels), over 280
paw-frames: the **mesh hallucinates 3.45 px of paw motion**, the tracker
0.02 px. But that comparison is near-tautological, and the decisive check
fails: during swing the tracker moves **0.05–0.28 px/frame against a true
~1.7**. It is locked to static background, not the paw.

Cause, measured not guessed: the refined-sole pixel is the lowest vertex of
the foot, so it lands **7–21 px BELOW the nearest actual dog pixel** — on the
floor, at the contact shadow, where texture is strong and static. A dense
grid confirms the model itself is fine (12 of 120 points moved, max 230 px);
only the seeds are wrong. Three seeding strategies were tried and all failed:
mesh-sole (frozen), mesh + upward foreground search (front legs only), and
silhouette-anchored via a temporal-median background (best visibility 0.79,
still 0.05–0.28 px/frame in swing). A paw here is small, low-texture,
self-occluding and motion-blurred — the hard case for point tracking.
Generic trackers need a correct seed and continuous appearance; neither is
available without semantic knowledge of what a paw is.

### Independence beats precision — the actual specification

With exact ground truth, identical solver and identical GT contacts, paw
pixels were corrupted two ways at matched magnitude:

| config (amp) | pixels | px err | root | toe | scale | traverse |
|---|---|---|---|---|---|---|
| h1.1_d4 (3.7) | CORRELATED (mesh) | 16.6 | 0.066 | 0.083 | −7.7% | 1.16 |
| h1.1_d4 | independent 8 px | 9.5 | **0.033** | 0.028 | −1.1% | 1.00 |
| h1.1_d4 | perfect | 0.0 | 0.034 | 0.026 | −1.6% | 1.02 |
| h0.8_d6 (7.5) | CORRELATED (mesh) | 14.6 | 0.118 | 0.137 | −6.5% | **1.82** |
| h0.8_d6 | independent 8 px | 9.5 | **0.044** | 0.040 | −2.3% | 1.07 |
| h0.8_d6 | perfect | 0.0 | 0.046 | 0.044 | −2.7% | 1.02 |
| h2.2_d2 (0.9) | CORRELATED (mesh) | 16.9 | 0.038 | 0.042 | +0.0% | 0.94 |
| h2.2_d2 | independent 8 px | 9.5 | 0.030 | 0.023 | +0.3% | 0.98 |

Read the rows at equal pixel error. **8 px of INDEPENDENT error is as good as
perfect pixels; ~16 px of mesh-CORRELATED error costs 2–3x.** Sweeping
independent noise 0→16 px moves root error only 0.034→0.039 m, so precision
past a few px buys nothing. The benefit grows with bad geometry — at
amplification 7.5 it fixes the traverse from +82% to +7%.

Why: mesh-derived pixels agree with the mesh's own FK error, so a wrong body
is self-consistent and the solver cannot see it. Independent pixels make the
inconsistency observable.

**This changes the roadmap item, not just its priority.** An independent 2D
paw detector is confirmed as the right next step, but for DECORRELATION, not
precision — and the spec is loose (~8 px), so DeepLabCut SuperAnimal-Quadruped
zero-shot may well suffice despite its documented "not for high-precision
use" caveat. Chasing sub-pixel tracking was the wrong target.

Caveat: contacts were held at ground truth throughout, so this isolates
placement. Better pixels should also improve contact detection, which is not
included in these numbers.

### DLC SuperAnimal-Quadruped, zero-shot — tried, does NOT beat the mesh

Installed in its own env (`animer/paw/dlcenv`, subprocess boundary);
`video2go2/paw_detect_dlc.py` + `paw_smoke_test.py`.

Detection quality is fine on dogs: 39 keypoints, dog_1 95.6% coverage at 0.79
mean confidence, dog_2 86.5% at 0.71, and the leg assignment solved
GEOMETRICALLY against the mesh reproduced the name-based mapping exactly (no
transposition). **cat_1 fails zero-shot** — 0.27 confidence, 48% coverage,
and the Hungarian assignment matched "RR" to `throat_base`, i.e. no plausible
back-right paw existed. The geometric cross-check exists to catch exactly
that and did.

End to end through the unmodified retargeter, same frames, same solver, only
the paw pixels swapped:

| clip | mesh pixels | DLC raw | DLC offset-corrected |
|---|---|---|---|
| dog_1 | **0.63%** clamp | 1.01% | 0.88% |
| dog_2 | **6.74%** | 9.79% | 8.56% |

DLC has a real SYSTEMATIC landmark offset — its keypoint is the paw centre,
not the sole — of 11-25 px, and strikingly reproducible across clips
(FR +10.3/+10.8, FL +11.2/+12.6, RR -4.4/-4.9 px on dog_1/dog_2). Removing it
per leg recovers part of the loss but does not reach the mesh.

**Why the harness prediction did not transfer.** The harness modelled
independent noise as WHITE. Measured on real video with duplicate frames
removed, DLC's static-window lag-1 autocorrelation is 0.85-0.89 — the same as
the mesh's 0.86-0.89. Its errors are decorrelated from the FK fit but still
strongly correlated IN TIME, so they do not average down over a stance run,
which was the mechanism the 2-3x gain depended on. The "8 px independent is
as good as perfect" result stands as stated; real detectors just do not
produce white error.

Not disproven: a FINE-TUNED detector, or one whose landmark is defined as the
sole. Both need hand-labelled frames, which remains the only real ground
truth for real footage.

### DLC full-skeleton pose refinement — marginal, watch the traverse

`video2go2/pose_refine_dlc.py`. SMPLify-style: refines AniMer's per-frame SMAL
pose against DLC's 2D skeleton (22 correspondences), with a prior holding it
near AniMer (DLC is 2D only, so depth must come from the prior) and temporal
smoothing. Reprojection 24.4 -> 20.8 px over 807 frames.

The SMAL-26 <-> DLC-39 correspondence was established EMPIRICALLY, not from
names, and the names would have wired it wrong:

    DLC front_*_thai -> elbow     DLC back_*_thai -> stifle (true knee)
    DLC front_*_knee -> WRIST     DLC back_*_knee -> HOCK
    DLC *_paw        -> paw       (12-24 px agreement with AniMer's skeleton)

That agreement is an independent cross-check on both models, and it settles
the "DLC puts paws at the knees" question: it does not — its naming is just
anatomically loose.

End to end on dog_1: clamp **0.63% -> 0.52%** (the first improvement measured
from any of the paw/skeleton work) BUT traverse fell 4.98 -> 4.35 m against a
baseline of 6.04 m. Per the motionless-robot trap, a clamp gain paid for with
a shrinking trajectory is not a win. Net: not adopted; needs the landmark
offsets solved as parameters rather than the pose absorbing them.

### ZoeDepth ground plane — good ORIENTATION, does not settle SCALE

`video2go2/depth_ground.py`: metric depth -> back-project with the CALIBRATED
focal -> RANSAC plane -> compare with the clicked calibration.

| clip | clicked height | ZoeDepth | ratio | normal disagreement |
|---|---|---|---|---|
| dog_1 | 1.101 m | 1.297 m | **+18%** | 5.2 deg |
| cat_1 | 1.090 m | 1.248 m | **+14%** | 2.3 deg |
| dog_2 | 1.143 m | 0.889 m | **-22%** | 3.9 deg |

Plane ORIENTATION agrees to 2-5 deg — good enough that the clicked geometry
could plausibly be replaced for shape. ABSOLUTE SCALE does not agree, and
disagrees in DIFFERENT DIRECTIONS, so ZoeDepth is a third opinion with its own
bias rather than an arbiter. Note dog_1 and cat_1 share a rig: the clicked
values agree to 1 cm and ZoeDepth's agree to 5 cm, both internally consistent
but ~15% apart from each other — so one of them carries a systematic bias in
that corridor and this test cannot say which. dog_2 is Veo-generated, so a
depth model there is out of domain and its -22% is the least trustworthy row.

**Conclusion: cannot discard the plane calibration for metric scale.** An
absolute ruler is still required, and the cheapest one is already in the
scene — dog_1's floor is covered in pens and markers of standard size, lying
flat ON the calibrated plane (so no height ambiguity, no d/h amplification).
That measurement is now the decisive open item.

### Size prior + ZoeDepth plane — the first real convergence

Scale is now a 1-D MAP in log space over three estimators plus a weak
BIOLOGICAL prior (shoulder 0.50 m, log-sigma 0.30 -> ~27-90 cm). Generic:
no fine-tuning, no per-clip tuning, true of any dog. `--size-prior`.

`depth_calib.py` builds a full calibration from ZoeDepth alone — no clicking,
no tile-size assumption — by fitting a plane and running Zhang's
factorisation backwards.

| clip / plane | scale | shoulder | traverse | clamp | skate |
|---|---|---|---|---|---|
| dog_1 clicked | 1.085 | 55.2 cm | 4.37 m | 0.51% | 0.011 |
| dog_1 **ZoeDepth** | 1.086 | 53.7 cm | 4.78 m | 1.06% | 0.016 |
| dog_1 baseline | 1.058 | 54.7 cm | **6.50 m** | 0.06% | 0.000 |
| dog_2 clicked | 1.191 | 62.2 cm | 8.86 m | 9.61% | 0.217 |
| dog_2 ZoeDepth | 0.787 | **40.8 cm** | 4.29 m | 6.01% | 0.081 |
| dog_2 baseline | 1.141 | 62.2 cm | 8.61 m | 1.78% | 0.031 |

**CORRECTION — the prior is nearly inert.** An ablation (which should have
been run before claiming anything) shows it carries 6-18% of the weight and
moves the answer by at most 4.6%:

| case | prior weight | MAP with | without | delta |
|---|---|---|---|---|
| dog_1 clicked | 6.2% | 1.085 | 1.092 | -0.7% |
| dog_2 clicked | 16.7% | 1.190 | 1.243 | -4.3% |
| dog_2 zoe | 18.2% | 0.789 | 0.754 | +4.6% |

It did NOT "kill" the 161 cm floor-spread estimate — that estimator only ever
had 1.6-4.5% weight because of the sigma=0.60 I assigned it by hand. The
MAP is really "lateral-anchor (64-88% of the weight) plus small corrections",
and every sigma in it is asserted by me, not measured. The prior mean of
0.50 m was also chosen after seeing that these clips imply 54-62 cm.

So this is a weighted opinion with hand-picked weights, not a measurement.
To make it sound: calibrate each estimator's sigma from its ACTUAL error
distribution on the 12 harness configs (which is what the harness is for),
and replace the Gaussian pull with a ONE-SIDED barrier that only penalises
biologically absurd sizes — the guard-rail role is the one the prior can
actually justify.

**dog_1 is now robust to the plane.** Two fully independent ground planes —
clicked tiles vs ZoeDepth metric depth — give scale 1.085 vs 1.086 and
shoulder 55.2 vs 53.7 cm. Traverse 4.37 vs 4.78 m, 9% apart. Both disagree
with the baseline's 6.50 m. That is not proof, but two independent
calibrations agreeing with each other and not with the baseline is the
strongest evidence yet on the traverse question — and it points the other way
from what I assumed for most of this work.

**A clean demonstration that clamp/skate cannot arbitrate**: dog_2's ZoeDepth
run scores BETTER on both (6.01% / 0.081 vs 9.61% / 0.217) while shrinking
the dog to 41 cm and halving the traverse. Exactly the motionless-robot trap.
Reprojection error is the only ground-truth-free metric here that punishes
collapse rather than rewarding it, and it should be the primary number.

### ZoeDepth vs clicked plane, prior disabled — the pattern

| clip | clicked | ZoeDepth | disagreement | footage |
|---|---|---|---|---|
| dog_1 | 1.0933 (56.1 cm) | 1.0912 (54.2 cm) | **0.2%** | real |
| cat_1 | 0.5115 (35.4 cm) | 0.5563 (38.9 cm) | **8.8%** | real |
| dog_2 | 1.2467 (67.7 cm) | 0.7533 (40.6 cm) | **65%** | Veo-generated |

ZoeDepth tracks the clicked calibration on REAL footage and diverges wildly on
the AI-generated clip. That makes a depth-derived plane a viable default for
real video — it removes the clicking step AND the tile-size assumption, which
is the genericity win — provided it is guarded by a rejection test rather than
adopted blindly. The biological size constraint is the right guard: not a
Gaussian nudging the estimate (measured inert), but a one-sided check that
REJECTS a plane implying an impossible animal, as dog_2's 41 cm retriever.

Caveats: the two real clips are the same corridor and rig, so this is really
n=1 scene. And cat_1's implied 35-39 cm shoulder is tall for a domestic cat
(~25 cm typical) under BOTH calibrations, hinting that they may share a
common over-scaling of that corridor — which only an absolute ruler (the pens
on dog_1's floor) can settle.

### dog_1.mov is 24 fps content in a 60 fps container

Measured at FULL resolution inside the dog's own bounding box: 61% of
consecutive frame pairs change by <0.05 grey levels, against 1.5-6 when the
dog moves. 1 - 24/60 = 60%. Frames are not bit-identical (re-encoding), so a
naive equality check misses it. Every velocity computed at 60 fps therefore
alternates between zero and ~2.5x true, which biases contact thresholds and
dilutes measured skate.

### Smoke test (`paw_smoke_test.py`) — reusable, and two traps it exposed

Judges any paw-pixel source with no ground truth: must be still where the raw
pixels are provably still, and must MOVE where they move. It correctly
rejects CoTracker (frozen on 3 of 4 legs) and flags the mesh's own drift.
Two calibration traps found while building it, both of which would have
condemned a good source:

* **dog_1.mov is frame-rate converted** — ~41% of frames are duplicates of
  their predecessor (60 fps container, ~36 fps of real content). A per-frame
  detector returns an identical answer on an identical frame, so per-frame
  displacement is bimodal and its MEDIAN sits in the duplicate population:
  DLC read as 0.38 px/frame ("frozen") versus 1.8 px/frame over a 2-frame
  stride. Measure motion over a stride. This also biases every speed-based
  contact detector in the pipeline and is worth revisiting there.
* **Autocorrelation cannot be read in a static window.** If the input frames
  are near-identical, any deterministic estimator returns a near-constant
  answer, whose autocorrelation is ~1 by construction regardless of quality.

## Not done

- Independent 2D paw detector — now SPECIFIED, not speculative (see the
  paw-measurement section): needs ~8 px accuracy and error INDEPENDENT of the
  mesh; sub-pixel precision is worthless. Try DeepLabCut SuperAnimal-Quadruped
  zero-shot first, in its own env behind a subprocess boundary. A point
  tracker was tried and rejected — do not re-attempt.
- The along-view body-height regression at the retargeter (see above) — the
  one thing blocking the BA path from replacing the baseline outright.
- ~~Full-AniMer runs on the harness renders~~ DONE (2026-07-29). On
  `cfg_h0.8_d4` (82.9% detection): real pose error is 8.4 cm median after
  scale normalization vs the injected 3.8 cm — the harness was ~2×
  optimistic FOR RENDERED APPEARANCE (the render is out-of-domain: beta
  wobble 17.8% vs 2.1% on real footage; real-footage pose error remains the
  measured 2-5 cm). Error is 91% below 3 Hz — the band-limited drift model
  is right in character. The depth cue overestimates depth on the render by
  the same ~40% as on real dog_1 (bias 0.71 vs 0.74) — same DIRECTION in
  both domains, but the solver-fitted bias on cat/dog_2 is ~0.93-0.96, so
  the bias varies per clip/animal and the cue stays relative-only. Fully
  real end-to-end at amp 5.0: lateral-anchor scale within 4% of truth
  (floor-spread would have been +93%), root error 16 cm, traverse ratio
  1.37 (inflated by the 17% detection gaps). `cfg_h2.2_d2` unusable — the
  DETECTOR finds the rendered dog in only 19% of frames from the high
  viewpoint; a more realistic render (texture/fur) is the fix.
- Per-frame root uncertainty from the solver Jacobian into the `.video.json`
  sidecar (PLAN.md Phase 4).
- Migration into `animal2go2` proper. Deliberate: it stays untouched until
  this path is trusted.
- The last 2.5 s of dog_1 (dog turns away, 52% of frames unanchored) is
  carried but weak; `--trim 0,660` drops it (the BA runs use it).
