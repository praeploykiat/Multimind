"""Quick comparison of trailing-buffer sizes on discussion-phase Whisper
transcripts only (skips audEERING - not needed to judge word accuracy),
to pick a value that fixes the shift/truncation issue without adding more
overlap bleed than necessary."""
import json
import os
import sys

import soundfile as sf
import whisper

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
GAME_VIDEO_NAME = "ONE NIGHT ULTIMATE WEREWOLF  Retro 3"
GAME_ID = "Game3"
AUDIO_PATH = os.path.join(RAW_DIR, "game3_audio.wav")

BUFFER_VALUES = [0.3, 0.45, 0.6]


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
    raise ValueError("Game not found")


def build_windows(dialogue, game_duration, buffer_sec):
    times = [to_sec(d["timestamp"]) for d in dialogue]
    windows = []
    for i, d in enumerate(dialogue):
        start = times[i]
        next_start = times[i + 1] if i + 1 < len(dialogue) else game_duration
        end = min(next_start + buffer_sec, game_duration)
        end = max(end, start + 0.5)
        windows.append((start, end))
    return windows


def main():
    game, _ = load_game_dialogue()
    dialogue = game["Dialogue"]
    duration = to_sec(game["endTime"]) - to_sec(game["startTime"])
    audio, sr = sf.read(AUDIO_PATH, dtype="float32")
    assert sr == 16000

    focus_ids = {32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51}

    print("Loading Whisper (base)...")
    model = whisper.load_model("base")

    results = {b: {} for b in BUFFER_VALUES}
    for buf in BUFFER_VALUES:
        windows = build_windows(dialogue, duration, buf)
        for d, (start, end) in zip(dialogue, windows):
            if d["Rec_Id"] not in focus_ids:
                continue
            seg = audio[int(start * sr):int(end * sr)]
            if seg.size < sr * 0.3:
                results[buf][d["Rec_Id"]] = ""
                continue
            r = model.transcribe(seg, language="en", fp16=False)
            results[buf][d["Rec_Id"]] = r["text"].strip()

    for d in dialogue:
        rid = d["Rec_Id"]
        if rid not in focus_ids:
            continue
        print(f"\nRec {rid} | GT: {d['utterance']!r}")
        for buf in BUFFER_VALUES:
            print(f"  buffer={buf}: {results[buf][rid]!r}")


if __name__ == "__main__":
    main()
