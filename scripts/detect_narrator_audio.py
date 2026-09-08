"""
Detect app-narrator audio bleed in the extracted per-utterance audio
segments, using a two-stage approach:

1. Keyword seeding: flag utterances whose Whisper transcript contains
   phrases characteristic of the ONUW night-phase narration script
   (rulebook-standard wording like "wake up", "close your eyes") - these
   are virtually never said by players during discussion.
2. Voice-embedding propagation: extract a speaker embedding (ECAPA-TDNN,
   Desplanques et al., INTERSPEECH 2020) for every utterance's audio
   segment, average the keyword-seeded segments into a reference
   "narrator voiceprint", then flag any OTHER segment whose embedding is
   highly similar to that reference - this catches narrator bleed even
   where Whisper's transcript doesn't contain a recognizable keyword.

This addresses a real gap in the MultiMind (ACMMM 2025) / OSUM pipeline,
which extracts audio purely by timestamp with no narrator handling.
"""
import json
import os

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
FEATURES_PATH = os.path.join(RAW_DIR, "game3_audio_features.json")
AUDIO_PATH = os.path.join(RAW_DIR, "game3_audio.wav")
OUTPUT_PATH = os.path.join(RAW_DIR, "game3_audio_features_with_narrator_flag.json")

# Rulebook-standard night-phase narration wording - not exact app script,
# but anchor phrases no player would organically say during discussion.
NARRATOR_KEYWORDS = [
    "wake up",
    "your eyes",  # catches "close your eyes" even when ASR drops "close"
    "look for other were",  # catches "werewolves" ASR mangling too
    "look at another",
    "look at your card",
    "exchange your card",
    "exchange cards between",
    "you may look",  # drop the trailing "at" requirement
    "you may exchange",
    "center car",  # catches "center card(s)" ASR mangling
]

SIMILARITY_THRESHOLD = 0.75  # cosine similarity to the narrator reference embedding


def is_keyword_seed(transcript):
    t = transcript.lower()
    return any(kw in t for kw in NARRATOR_KEYWORDS)


def main():
    with open(FEATURES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    utterances = data["utterances"]

    audio, sr = sf.read(AUDIO_PATH, dtype="float32")
    assert sr == 16000

    print("Loading ECAPA-TDNN speaker embedding model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(RAW_DIR, "pretrained_ecapa"),
    )

    embeddings = {}
    for u in utterances:
        start, end = u["window_sec"]
        seg = audio[int(start * sr):int(end * sr)]
        if seg.size < sr * 0.3:
            embeddings[u["Rec_Id"]] = None
            continue
        wav_tensor = torch.from_numpy(seg).unsqueeze(0)
        with torch.no_grad():
            emb = classifier.encode_batch(wav_tensor).squeeze().numpy()
        embeddings[u["Rec_Id"]] = emb / (np.linalg.norm(emb) + 1e-8)

    seed_ids = [u["Rec_Id"] for u in utterances if is_keyword_seed(u.get("asr_transcript", ""))]
    seed_embs = [embeddings[rid] for rid in seed_ids if embeddings[rid] is not None]
    print(f"Keyword-seeded narrator segments: {seed_ids}")
    assert seed_embs, "No keyword-seeded segments found - can't build a reference voiceprint"

    narrator_ref = np.mean(seed_embs, axis=0)
    narrator_ref /= np.linalg.norm(narrator_ref) + 1e-8

    for u in utterances:
        emb = embeddings[u["Rec_Id"]]
        if emb is None:
            u["narrator_similarity"] = None
            u["narrator_flag"] = False
            continue
        sim = float(np.dot(emb, narrator_ref))
        u["narrator_similarity"] = round(sim, 4)
        u["narrator_flag"] = bool(u["Rec_Id"] in seed_ids or sim >= SIMILARITY_THRESHOLD)

    n_flagged = sum(1 for u in utterances if u["narrator_flag"])
    print(f"\nFlagged {n_flagged}/{len(utterances)} utterances as narrator-contaminated:")
    for u in utterances:
        mark = "NARRATOR" if u["narrator_flag"] else "clean"
        print(f"  [{mark:8s}] Rec_Id {u['Rec_Id']:3d} {u['timestamp']} sim={u['narrator_similarity']} "
              f"GT={u['utterance_ground_truth']!r}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
