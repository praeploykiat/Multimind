"""
Diagnostic: overlay the FULL gaze heatmap (not just its peak point) for one
frame, to check whether Gaze-LLE is genuinely confident about a precise
target or diffusely uncertain - the arrow-only rendering in
annotate_gaze_frames.py can make a low-confidence, spread-out prediction
look like a precise "looking at that face" claim when it isn't.
"""
import json
import os
import sys

import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from gazelle.model import get_gazelle_model

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
FRAME_DIR = os.path.join(RAW_DIR, "gaze_frames")
OUT_DIR = os.path.join(RAW_DIR, "gaze_heatmap_diagnostic")
FACE_MODEL_PATH = os.path.join(RAW_DIR, "blaze_face_short_range.tflite")
GAZE_CKPT_PATH = os.path.join(RAW_DIR, "gazelle_dinov2_vitb14_inout.pt")

REC_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 34


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frame_path = os.path.join(FRAME_DIR, f"frame_{REC_ID:03d}.jpg")

    face_detector = mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            min_detection_confidence=0.3,
        )
    )
    model, transform = get_gazelle_model("gazelle_dinov2_vitb14_inout")
    model.load_gazelle_state_dict(torch.load(GAZE_CKPT_PATH, map_location="cpu"))
    model.eval()

    mp_img = mp.Image.create_from_file(frame_path)
    det_result = face_detector.detect(mp_img)
    pil_img = Image.open(frame_path).convert("RGB")
    w, h = pil_img.size

    bboxes = []
    for det in det_result.detections:
        bb = det.bounding_box
        bboxes.append((bb.origin_x / w, bb.origin_y / h,
                        (bb.origin_x + bb.width) / w, (bb.origin_y + bb.height) / h))

    input_data = {"images": transform(pil_img).unsqueeze(0), "bboxes": [bboxes]}
    with torch.no_grad():
        output = model(input_data)

    for face_idx, bbox in enumerate(bboxes):
        heatmap = output["heatmap"][0][face_idx].numpy()  # [64,64], values in [0,1]
        inout = output["inout"][0][face_idx].item() if output["inout"] is not None else None

        # normalize heatmap to 0-255 and resize to full image
        hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        hm_img = Image.fromarray((hm_norm * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        hm_rgba = Image.new("RGBA", (w, h))
        hm_arr = np.array(hm_img)
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[..., 0] = 255  # red channel
        overlay[..., 3] = (hm_arr * 0.75).astype(np.uint8)  # alpha follows heatmap intensity
        hm_rgba = Image.fromarray(overlay, mode="RGBA")

        base = pil_img.convert("RGBA")
        composited = Image.alpha_composite(base, hm_rgba).convert("RGB")

        from PIL import ImageDraw
        draw = ImageDraw.Draw(composited)
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle((xmin * w, ymin * h, xmax * w, ymax * h), outline="lime", width=3)
        draw.text((xmin * w, max(0, ymin * h - 24)),
                   f"P{face_idx} inout={inout:.2f} max={heatmap.max():.3f} spread={hm_norm.mean():.3f}",
                   fill="lime")

        out_path = os.path.join(OUT_DIR, f"frame_{REC_ID:03d}_face{face_idx}.jpg")
        composited.save(out_path, quality=90)
        print(f"face {face_idx}: inout={inout:.3f} heatmap_max={heatmap.max():.4f} "
              f"heatmap_mean={heatmap.mean():.4f} (higher mean relative to max = more diffuse/uncertain)")

    print(f"\nSaved diagnostic overlays to {OUT_DIR}")


if __name__ == "__main__":
    main()
