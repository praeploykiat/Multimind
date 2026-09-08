"""
Extract per-utterance audio features (transcription + vocal emotion) from a
Werewolf-Among-Us game recording.

Transcription: openai/whisper (Radford et al., ICML 2023)
Vocal emotion (arousal/valence/dominance): audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
    (Wagner et al., IEEE TPAMI 2023)
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import whisper
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel


class RegressionHead(nn.Module):
    """Classification head, as defined by audeering's model card."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    """Speech emotion regressor (arousal/dominance/valence), matching the
    architecture audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim was
    actually trained with. The generic Wav2Vec2ForSequenceClassification
    head does NOT match this checkpoint's parameter names, so loading it
    that way silently drops the trained head and substitutes a randomly
    initialized one."""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        hidden_states = self.wav2vec2(input_values)[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        return self.classifier(hidden_states)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
GAME_VIDEO_NAME = "ONE NIGHT ULTIMATE WEREWOLF  Retro 3"
GAME_ID = "Game3"
AUDIO_PATH = os.path.join(RAW_DIR, "game3_audio.wav")
OUTPUT_PATH = os.path.join(RAW_DIR, "game3_audio_features.json")

EMOTION_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
WHISPER_MODEL_SIZE = "base"


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


TRAILING_BUFFER_SEC = 0.6  # see build_utterance_windows docstring


def build_utterance_windows(dialogue, game_duration):
    """Each utterance's window runs from its own timestamp to the next
    utterance's timestamp (or end of game for the last one), extended by
    TRAILING_BUFFER_SEC.

    Timestamps mark when an utterance was logged, not necessarily the exact
    audio onset - comparing transcripts against ground truth showed a
    recurring "off by one" pattern (e.g. Rec_Id 41-43 in game3: the words
    for each utterance kept showing up in the FOLLOWING utterance's window
    instead), consistent with actual speech onset lagging behind its
    timestamp. Cutting exactly at the next timestamp then clips it into the
    wrong segment. The trailing buffer intentionally overlaps a bit into the
    next utterance's own window rather than risk cutting speech off - some
    duplicate audio between adjacent segments is a smaller problem than
    misattributing whole utterances downstream.
    """
    times = [to_sec(d["timestamp"]) for d in dialogue]
    windows = []
    for i, d in enumerate(dialogue):
        start = times[i]
        next_start = times[i + 1] if i + 1 < len(dialogue) else game_duration
        end = min(next_start + TRAILING_BUFFER_SEC, game_duration)
        end = max(end, start + 0.5)  # guard against zero-length windows
        windows.append((start, end))
    return windows


def load_audio_segment(waveform, sample_rate, start_sec, end_sec):
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    return waveform[:, start_sample:end_sample]


def main():
    game, split = load_game_dialogue()
    duration = to_sec(game["endTime"]) - to_sec(game["startTime"])
    dialogue = game["Dialogue"]
    windows = build_utterance_windows(dialogue, duration)
    print(f"Loaded {game['video_name']} / {game['Game_ID']} from {split} split: "
          f"{len(dialogue)} utterances, {duration}s")

    print("Loading audio...")
    audio_np, sample_rate = sf.read(AUDIO_PATH, dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    assert sample_rate == 16000, f"expected 16kHz audio, got {sample_rate}"

    print(f"Loading Whisper ({WHISPER_MODEL_SIZE})...")
    asr_model = whisper.load_model(WHISPER_MODEL_SIZE)

    print(f"Loading emotion model ({EMOTION_MODEL_ID})...")
    emo_processor = Wav2Vec2Processor.from_pretrained(EMOTION_MODEL_ID)
    emo_model = EmotionModel.from_pretrained(EMOTION_MODEL_ID)
    emo_model.eval()

    results = []
    for i, (d, (start, end)) in enumerate(zip(dialogue, windows)):
        seg = load_audio_segment(waveform, sample_rate, start, end)
        seg_np = seg.squeeze(0).numpy()

        item = {
            "Rec_Id": d["Rec_Id"],
            "speaker": d["speaker"],
            "timestamp": d["timestamp"],
            "window_sec": [round(start, 2), round(end, 2)],
            "utterance_ground_truth": d["utterance"],
            "annotation": d["annotation"],
        }

        if seg_np.size < sample_rate * 0.3:  # skip near-empty slivers
            item["asr_transcript"] = ""
            item["arousal"] = item["valence"] = item["dominance"] = None
            results.append(item)
            continue

        # --- Transcription (Whisper) ---
        asr_result = asr_model.transcribe(seg_np.astype(np.float32), language="en", fp16=False)
        item["asr_transcript"] = asr_result["text"].strip()

        # --- Vocal emotion (audEERING wav2vec2-robust) ---
        # processor normalizes the raw waveform; model expects (batch, samples)
        y = emo_processor(seg_np, sampling_rate=sample_rate)
        y = torch.from_numpy(y["input_values"][0]).reshape(1, -1)
        with torch.no_grad():
            logits = emo_model(y).squeeze(0).tolist()
        # model output order confirmed from the model card's own example: arousal, dominance, valence
        item["arousal"], item["dominance"], item["valence"] = logits

        results.append(item)
        print(f"[{i+1}/{len(dialogue)}] {d['speaker']}: "
              f"asr='{item['asr_transcript'][:40]}' "
              f"a={item['arousal']:.2f} v={item['valence']:.2f} d={item['dominance']:.2f}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"game": {"video_name": game["video_name"], "Game_ID": game["Game_ID"],
                             "split": split, "duration_sec": duration},
                   "utterances": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} utterance records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
