"""Run AniMer on a video and render the predicted SMAL mesh over every frame."""
from pathlib import Path
import argparse
import contextlib
import io
import os
import subprocess

import cv2
import numpy as np
import torch
import torch.utils.data
import trimesh
import pyrender
import detectron2.config
import detectron2.engine
from detectron2 import model_zoo

from amr.models import load_amr
from amr.utils import recursive_to
from amr.datasets.vitdet_dataset import ViTDetDataset
from amr.utils.renderer import Renderer, cam_crop_to_full, create_raymond_lights

import warnings
warnings.filterwarnings("ignore")

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
# COCO contiguous ids for cat, dog, horse, sheep, cow, bear, zebra
ANIMAL_CLASSES = [15, 16, 17, 18, 19, 21, 22]


class VideoRenderer:
    """Renderer that keeps a single offscreen GL context alive across frames.

    amr.utils.renderer.Renderer builds and tears down an OffscreenRenderer on
    every call, which dominates runtime over thousands of frames.
    """

    def __init__(self, renderer: Renderer, width: int, height: int):
        self.renderer = renderer
        self.offscreen = pyrender.OffscreenRenderer(viewport_width=width,
                                                    viewport_height=height,
                                                    point_size=1.0)
        self.width = width
        self.height = height

    def render_rgba(self, vertices, cam_t, focal_length,
                    mesh_base_color=LIGHT_BLUE, scene_bg_color=(0, 0, 0)):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.7, alphaMode='OPAQUE',
            baseColorFactor=(*mesh_base_color, 1.0))
        meshes = [pyrender.Mesh.from_trimesh(
            self.renderer.vertices_to_trimesh(v, t.copy(), mesh_base_color),
            material=material)
            for v, t in zip(vertices, cam_t)]

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        for i, mesh in enumerate(meshes):
            scene.add(mesh, f'mesh_{i}')

        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=self.width / 2., cy=self.height / 2.,
                                           zfar=1e12)
        camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
        scene.add_node(camera_node)
        self.renderer.add_point_lighting(scene, camera_node)
        self.renderer.add_lighting(scene, camera_node)
        for node in create_raymond_lights():
            scene.add_node(node)

        color, _ = self.offscreen.render(scene, flags=pyrender.RenderFlags.RGBA)
        return color.astype(np.float32) / 255.0

    def delete(self):
        self.offscreen.delete()


def build_detector(score_thresh: float):
    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
    return detectron2.engine.DefaultPredictor(cfg)


def detect_animals(detector, img_rgb, score_thresh, max_animals):
    instances = detector(img_rgb)['instances']
    keep = [i for i, (c, s) in enumerate(zip(instances.pred_classes, instances.scores))
            if (int(c) in ANIMAL_CLASSES) and (float(s) > score_thresh)]
    if not keep:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = instances.pred_boxes.tensor[keep].cpu().numpy()
    scores = instances.scores[keep].cpu().numpy()
    if max_animals > 0 and len(boxes) > max_animals:
        boxes = boxes[np.argsort(-scores)[:max_animals]]
    return boxes


def open_writer(path, width, height, fps, crf):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', f'{width}x{height}', '-r', f'{fps}',
           '-i', 'pipe:0',
           '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf),
           '-pix_fmt', 'yuv420p', str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(description='AniMer video demo')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--out_video', type=str, required=True)
    parser.add_argument('--max_side', type=int, default=1280,
                        help='Downscale frames so the longest side is at most this (0 = keep native)')
    parser.add_argument('--frame_stride', type=int, default=1,
                        help='Process every Nth frame; output fps is divided accordingly')
    parser.add_argument('--max_frames', type=int, default=0, help='0 = whole video')
    parser.add_argument('--det_thresh', type=float, default=0.7)
    parser.add_argument('--max_animals', type=int, default=1,
                        help='Keep only the N highest-scoring detections per frame (0 = all)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Mesh opacity in the overlay')
    parser.add_argument('--out_video_sbs', type=str, default=None,
                        help='Optional second output with the original and the overlay side by side')
    parser.add_argument('--crf', type=int, default=18)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, model_cfg = load_amr(args.checkpoint)
    model = model.to(device).eval()
    detector = build_detector(args.det_thresh)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f'Could not open {args.video}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = 1.0
    if args.max_side > 0 and max(src_w, src_h) > args.max_side:
        scale = args.max_side / max(src_w, src_h)
    # keep dimensions even for yuv420p
    width = int(round(src_w * scale)) // 2 * 2
    height = int(round(src_h * scale)) // 2 * 2

    base_renderer = Renderer(model_cfg, faces=model.smal.faces)
    video_renderer = VideoRenderer(base_renderer, width, height)

    out_fps = src_fps / max(1, args.frame_stride)
    Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)
    writer = open_writer(args.out_video, width, height, out_fps, args.crf)
    writer_sbs = None
    if args.out_video_sbs:
        Path(args.out_video_sbs).parent.mkdir(parents=True, exist_ok=True)
        writer_sbs = open_writer(args.out_video_sbs, width * 2, height, out_fps, args.crf)

    focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(width, height)

    processed = 0
    detected = 0
    idx = -1
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            idx += 1
            if idx % args.frame_stride != 0:
                continue
            if args.max_frames and processed >= args.max_frames:
                break

            if (width, height) != (src_w, src_h):
                frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            boxes = detect_animals(detector, frame_rgb, args.det_thresh, args.max_animals)

            all_verts, all_cam_t = [], []
            if len(boxes):
                dataset = ViTDetDataset(model_cfg, frame_rgb, boxes)
                loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
                # ViTDetDataset prints a debug line per crop; keep the log readable
                with contextlib.redirect_stdout(io.StringIO()):
                    batches = list(loader)
                for batch in batches:
                    batch = recursive_to(batch, device)
                    with torch.no_grad():
                        out = model(batch)
                    cam_t_full = cam_crop_to_full(out['pred_cam'],
                                                  batch['box_center'].float(),
                                                  batch['box_size'].float(),
                                                  batch['img_size'].float(),
                                                  focal_length).detach().cpu().numpy()
                    verts = out['pred_vertices'].detach().cpu().numpy()
                    all_verts.extend(list(verts))
                    all_cam_t.extend(list(cam_t_full))

            overlay = frame_bgr
            if all_verts:
                detected += 1
                rgba = video_renderer.render_rgba(all_verts, all_cam_t, focal_length)
                mask = rgba[:, :, 3:4] * args.alpha
                mesh_bgr = rgba[:, :, :3][:, :, ::-1] * 255.0
                overlay = (mesh_bgr * mask + frame_bgr.astype(np.float32) * (1 - mask))
                overlay = overlay.astype(np.uint8)

            writer.stdin.write(np.ascontiguousarray(overlay).tobytes())
            if writer_sbs is not None:
                sbs = np.concatenate([frame_bgr, overlay], axis=1)
                writer_sbs.stdin.write(np.ascontiguousarray(sbs).tobytes())

            processed += 1
            if processed % 25 == 0:
                total = n_frames // max(1, args.frame_stride) if n_frames > 0 else 0
                print(f'{Path(args.video).name}: {processed}/{total} frames, '
                      f'{detected} with a detected animal', flush=True)
    finally:
        cap.release()
        writer.stdin.close()
        writer.wait()
        if writer_sbs is not None:
            writer_sbs.stdin.close()
            writer_sbs.wait()
        video_renderer.delete()

    print(f'Done: {processed} frames written to {args.out_video} '
          f'({detected} had a detection, {processed - detected} passed through unchanged)')


if __name__ == '__main__':
    main()
