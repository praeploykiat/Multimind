"""
Extract per-utterance gaze features from a Werewolf-Among-Us game recording.

Face detection: MediaPipe BlazeFace (short-range)
Gaze target estimation: Gaze-LLE (Ryan et al., CVPR 2025 Highlight)
    https://github.com/fkryan/gazelle - frozen DINOv2 backbone + small
    learned decoder (~12MB), CPU-feasible.

Note: unlike Rec_Id/speaker in the text/audio pipelines, faces here are NOT
attributed to named players - that would need a seat/identity mapping this
script doesn't attempt (the MultiMind paper itself did this step manually,
by hand-cropping per-player regions, since players stay seated throughout).
Faces are reported by their detection index within each frame instead.

Setup (from werewolf_among_us/):
    git clone https://github.com/fkryan/gazelle.git gazelle_src
    touch gazelle_src/gazelle/__init__.py   # upstream repo is missing this
    pip install -e gazelle_src torchvision timm mediapipe
    curl -L -o raw/blaze_face_short_range.tflite \
        https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
    curl -L -o raw/gazelle_dinov2_vitb14_inout.pt \
        https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt
(the DINOv2 backbone itself auto-downloads from torch hub on first run, ~330MB)
"""
import json
import os

import mediapipe as mp
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

from gazelle.model import get_gazelle_model

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
GAME_VIDEO_NAME = "ONE NIGHT ULTIMATE WEREWOLF  Retro 3"
GAME_ID = "Game3"
VIDEO_PATH = os.path.join(RAW_DIR, "game3.mp4")
FRAME_DIR = os.path.join(RAW_DIR, "gaze_frames")
OUTPUT_PATH = os.path.join(RAW_DIR, "game3_gaze_features.json")

FACE_MODEL_PATH = os.path.join(RAW_DIR, "blaze_face_short_range.tflite")
GAZE_MODEL_NAME = "gazelle_dinov2_vitb14_inout"
GAZE_CKPT_PATH = os.path.join(RAW_DIR, "gazelle_dinov2_vitb14_inout.pt")
FACE_DET_CONFIDENCE = 0.3


def to_sec(t):
    parts = [int(p) for p in t.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def load_game_dialogue():
    for split in ["train", "val", "test"]:
        data = json.load(open(os.path.join(RAW_DIR, f"{split}.json"), encoding="utf-8"))
        for g in data:
            if g["video_name"] == GAME_VIDEO_NAME and g["Game_ID"] == GAME_ID:
                return g, split
    raise ValueError("Game not found in any split")


def extract_frame(timestamp_sec, out_path):
    os.system(f'ffmpeg -y -ss {timestamp_sec} -i "{VIDEO_PATH}" -vframes 1 -q:v 2 "{out_path}" -loglevel error')


def bbox_contains_point(bbox, x, y):
    xmin, ymin, xmax, ymax = bbox
    return xmin <= x <= xmax and ymin <= y <= ymax


def main():
    game, split = load_game_dialogue()
    dialogue = game["Dialogue"]
    print(f"Loaded {game['video_name']} / {game['Game_ID']} from {split} split: {len(dialogue)} utterances")

    os.makedirs(FRAME_DIR, exist_ok=True)

    face_detector = mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            min_detection_confidence=FACE_DET_CONFIDENCE,
        )
    )

    print(f"Loading Gaze-LLE ({GAZE_MODEL_NAME})...")
    gaze_model, transform = get_gazelle_model(GAZE_MODEL_NAME)
    gaze_model.load_gazelle_state_dict(torch.load(GAZE_CKPT_PATH, map_location="cpu"))
    gaze_model.eval()

    results = []
    for i, d in enumerate(dialogue):
        t = to_sec(d["timestamp"])
        frame_path = os.path.join(FRAME_DIR, f"frame_{d['Rec_Id']:03d}.jpg")
        extract_frame(t, frame_path)

        item = {
            "Rec_Id": d["Rec_Id"], "speaker": d["speaker"], "timestamp": d["timestamp"],
            "utterance_ground_truth": d["utterance"], "annotation": d["annotation"],
            "faces": [],
        }

        if not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
            results.append(item)
            continue

        mp_img = mp.Image.create_from_file(frame_path)
        det_result = face_detector.detect(mp_img)
        pil_img = Image.open(frame_path).convert("RGB")
        w, h = pil_img.size

        bboxes = []
        for det in det_result.detections:
            bb = det.bounding_box
            bboxes.append((bb.origin_x / w, bb.origin_y / h,
                            (bb.origin_x + bb.width) / w, (bb.origin_y + bb.height) / h))

        if not bboxes:
            results.append(item)
            print(f"[{i+1}/{len(dialogue)}] {d['speaker']}: no faces detected")
            continue

        input_data = {"images": transform(pil_img).unsqueeze(0), "bboxes": [bboxes]}
        with torch.no_grad():
            output = gaze_model(input_data)

        for face_idx, bbox in enumerate(bboxes):
            heatmap = output["heatmap"][0][face_idx]
            inout = output["inout"][0][face_idx].item() if output["inout"] is not None else None
            peak_idx = heatmap.argmax().item()
            py, px = peak_idx // heatmap.shape[1], peak_idx % heatmap.shape[1]
            gaze_x, gaze_y = px / heatmap.shape[1], py / heatmap.shape[0]

            looking_at_face = None
            for other_idx, other_bbox in enumerate(bboxes):
                if other_idx != face_idx and bbox_contains_point(other_bbox, gaze_x, gaze_y):
                    looking_at_face = other_idx
                    break

            item["faces"].append({
                "face_idx": face_idx,
                "bbox": [round(v, 3) for v in bbox],
                "in_frame_score": round(inout, 3) if inout is not None else None,
                "gaze_target_xy": [round(gaze_x, 3), round(gaze_y, 3)],
                "looking_at_face_idx": looking_at_face,
            })

        results.append(item)
        summary = ", ".join(f"f{fc['face_idx']}->f{fc['looking_at_face_idx']}"
                             if fc["looking_at_face_idx"] is not None else f"f{fc['face_idx']}->none"
                             for fc in item["faces"])
        print(f"[{i+1}/{len(dialogue)}] {d['speaker']}: {len(bboxes)} faces, gaze: {summary}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"game": {"video_name": game["video_name"], "Game_ID": game["Game_ID"], "split": split},
                   "utterances": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} utterance records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
