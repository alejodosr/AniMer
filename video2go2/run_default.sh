#!/usr/bin/env bash
# DEFAULT video -> Go2 pipeline. No manual ground-plane calibration.
#
#   usage:  video2go2/run_default.sh <video> <clip-name> [trim_start,trim_end]
#
# The ground plane comes from ZoeDepth metric depth, not from clicking four
# tile corners. That removes both the interactive step and the tile-size
# assumption that made the old calibration's metres provisional. Measured
# against the clicked calibration on real footage it agrees to 0.2% (dog_1)
# and 8.8% (cat_1); it diverges on AI-generated video (dog_2, 65%), which is
# the known limitation.
#
# Scale is a log-space combination of three estimators with NO biological
# prior: an ablation showed the prior carried 6-18% of the weight and moved
# the answer by at most 4.6%, while every sigma in it was hand-asserted.
#
# Stages:
#   1  animer_infer      SMAL pose per frame            (GPU, conda 'animal')
#   2  paw_detect_dlc    independent 2D skeleton        (GPU, dlcenv)
#   3  pose_refine_dlc   refine SMAL against DLC        (optional, see below)
#   4  depth_calib       ground plane from ZoeDepth     (GPU, dlcenv)
#   5  contacts_kine     stance/swing, no floor speed
#   6  world_place_ba    clip-wide placement, --size-prior 0
#   7  parse_video       Milestone 1 npz contract
#
# Stage 3 is skipped when DLC coverage is below --min-coverage: on cat_1 the
# detector only holds 48% of frames and refining against it is worse than not.
set -euo pipefail

VIDEO="${1:?usage: run_default.sh <video> <clip> [trim]}"
CLIP="${2:?}"
TRIM="${3:-}"

REPO=/home/alejodosr/py_workspace/AniMer
OUT=/media/SHARED_DATA/postcapitalistrobots/animer/v2go2/default
CALIB=$REPO/calib_depth
PY_ANIMAL=/home/alejodosr/anaconda3/envs/animal/bin/python
PY_DLC=/media/SHARED_DATA/postcapitalistrobots/animer/paw/dlcenv/bin/python
mkdir -p "$OUT" "$CALIB"
cd "$REPO"

TRIMARG=""
[ -n "$TRIM" ] && TRIMARG="--trim $TRIM"

echo "=== 1/7 AniMer pose ==="
[ -f "$OUT/${CLIP}_animer.npz" ] || \
  PYOPENGL_PLATFORM=egl FVCORE_CACHE=/media/SHARED_DATA/postcapitalistrobots/animer/detectron2_cache \
  $PY_ANIMAL video2go2/animer_infer.py --video "$VIDEO" --out "$OUT/${CLIP}_animer.npz"

echo "=== 2/7 DLC 2D skeleton ==="
[ -f "$OUT/${CLIP}_dlc_all.npz" ] || \
  PYTHONNOUSERSITE=1 $PY_DLC video2go2/paw_detect_dlc.py \
    --video "$VIDEO" --mesh "$OUT/${CLIP}_animer.npz" \
    --out "$OUT/${CLIP}_dlc.npz" --dest "$OUT/${CLIP}_dlc_raw"

echo "=== 3/7 ground plane from ZoeDepth (no clicking) ==="
[ -f "$CALIB/${CLIP}_depth.json" ] || \
  PYTHONNOUSERSITE=1 $PY_DLC video2go2/depth_calib.py \
    --video "$VIDEO" --ref-calib "$CALIB/${CLIP}_depth.json" \
    --out "$CALIB/${CLIP}_depth.json"

INFER="$OUT/${CLIP}_animer.npz"
echo "=== 4/7 optional pose refinement against DLC ==="
if $PY_ANIMAL video2go2/pose_refine_dlc.py --infer "$INFER" \
      --dlc "$OUT/${CLIP}_dlc_all.npz" --out "$OUT/${CLIP}_refined.npz"; then
  INFER="$OUT/${CLIP}_refined.npz"
fi

echo "=== 5/7 contacts ==="
$PY_ANIMAL video2go2/contacts_kine.py --infer "$INFER" \
  --calib "$CALIB/${CLIP}_depth.json" --out "$OUT/${CLIP}_contacts.npz"

echo "=== 6/7 world placement (BA, no size prior) ==="
$PY_ANIMAL video2go2/world_place_ba.py --infer "$INFER" \
  --contacts "$OUT/${CLIP}_contacts.npz" --calib "$CALIB/${CLIP}_depth.json" \
  --out "$OUT/${CLIP}_world.npz" --size-prior 0 $TRIMARG

echo "=== 7/7 Milestone 1 npz ==="
$PY_ANIMAL video2go2/parse_video.py --world "$OUT/${CLIP}_world.npz" \
  --source "$CLIP" --out "$OUT/processed/${CLIP}.npz"

echo
echo "done -> $OUT/processed/${CLIP}.npz"
echo "retarget with:  cd ../animal2go2 && .venv/bin/python retarget/retarget.py $OUT/processed/${CLIP}.npz"
