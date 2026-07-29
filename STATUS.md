# STATUS — video → Go2

Turning a monocular video of an animal into the npz that `animal2go2` already
retargets onto a Unitree Go2. Working end to end on three clips.

**animal2go2 is never modified.** Everything here lives in `video2go2/` and
produces a file that its unmodified Milestone 1 pipeline consumes.

---

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

## Not done

- Independent 2D paw detector (the measurement to justify it is specified above).
- Synthetic harness with exact ground truth. AniMer recovers a pose it
  generated to 3.6% of body length, so a rendered-mesh harness *is* viable —
  the earlier objection was based on stick-figure renders and is withdrawn.
- Migration into `animal2go2` proper. Deliberate: it stays untouched until
  this path is trusted.
- The last 2.5 s of dog_1 (dog turns away, 52% of frames unanchored) is
  carried but weak; `--trim 0,660` drops it.
