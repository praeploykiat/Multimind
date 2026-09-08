"""
Draw face boxes + gaze arrows onto the extracted frames, using the results
already saved by extract_gaze_features.py. Each face gets a distinct color;
an arrow points from a face's center to whatever it's looking at (another
face's center if looking_at_face_idx is set, otherwise the raw gaze point).
"""
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
FEATURES_PATH = os.path.join(RAW_DIR, "game3_gaze_features.json")
FRAME_DIR = os.path.join(RAW_DIR, "gaze_frames")
OUT_DIR = os.path.join(RAW_DIR, "gaze_frames_annotated")

COLORS = ["#e6194B", "#4363d8", "#3cb44b", "#ffe119", "#911eb4"]  # red, blue, green, yellow, purple
NOT_LOOKING_COLOR = "#999999"


def bbox_center(bbox, w, h):
    xmin, ymin, xmax, ymax = bbox
    return ((xmin + xmax) / 2 * w, (ymin + ymax) / 2 * h)


def draw_arrow(draw, start, end, color, width=3):
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_len = 12
    for side in (0.5, -0.5):
        hx = end[0] - head_len * math.cos(angle - side * math.pi / 4)
        hy = end[1] - head_len * math.sin(angle - side * math.pi / 4)
        draw.line([end, (hx, hy)], fill=color, width=width)


def main():
    data = json.load(open(FEATURES_PATH, encoding="utf-8"))
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    for item in data["utterances"]:
        frame_path = os.path.join(FRAME_DIR, f"frame_{item['Rec_Id']:03d}.jpg")
        if not os.path.exists(frame_path) or not item["faces"]:
            continue

        img = Image.open(frame_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)
        faces = item["faces"]
        centers = [bbox_center(f["bbox"], w, h) for f in faces]

        for i, face in enumerate(faces):
            color = COLORS[i % len(COLORS)]
            xmin, ymin, xmax, ymax = face["bbox"]
            box = (xmin * w, ymin * h, xmax * w, ymax * h)
            draw.rectangle(box, outline=color, width=3)
            label = f"P{i}"
            draw.text((box[0] + 2, max(0, box[1] - 24)), label, fill=color, font=font)

        for i, face in enumerate(faces):
            color = COLORS[i % len(COLORS)]
            start = centers[i]
            target_idx = face["looking_at_face_idx"]
            if target_idx is not None:
                end = centers[target_idx]
                draw_arrow(draw, start, end, color)
            else:
                gx, gy = face["gaze_target_xy"]
                end = (gx * w, gy * h)
                draw_arrow(draw, start, end, NOT_LOOKING_COLOR)

        out_path = os.path.join(OUT_DIR, f"frame_{item['Rec_Id']:03d}.jpg")
        img.save(out_path, quality=90)

    print(f"Annotated frames saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
