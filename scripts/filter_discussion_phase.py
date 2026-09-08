"""
Filter extracted audio (or gaze) features down to the discussion phase only,
dropping night-phase and pre-discussion small talk - both of which risk
app-narrator audio contamination (see detect_narrator_audio.py exploration)
and aren't the phenomenon of interest for persuasion-strategy analysis
anyway.

Rule (generalizes across all 199 games, no per-game timestamp guessing):
keep every utterance from the FIRST one in the game whose annotation is not
['No Strategy'] onward. Before that point, players are either in the night
phase (silent role-actions narrated by the app) or exchanging pre-discussion
small talk - the dataset's own persuasion-strategy annotation confirms
nothing strategically meaningful happens yet.
"""
import json
import os
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")


def first_discussion_index(dialogue_or_utterances, annotation_key="annotation"):
    for i, u in enumerate(dialogue_or_utterances):
        ann = u.get(annotation_key, [])
        if ann and ann != ["No Strategy"]:
            return i
    return None


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAW_DIR, "game3_audio_features.json")
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path.replace(".json", "_discussion_only.json")

    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)

    utterances = data["utterances"]
    idx = first_discussion_index(utterances)
    if idx is None:
        print("No discussion-phase utterance found (all 'No Strategy') - nothing to keep")
        return

    dropped = utterances[:idx]
    kept = utterances[idx:]
    data["utterances"] = kept
    data["discussion_phase_filter"] = {
        "rule": "first utterance with annotation != ['No Strategy'], onward",
        "dropped_count": len(dropped),
        "kept_count": len(kept),
        "cutoff_rec_id": kept[0]["Rec_Id"],
        "cutoff_timestamp": kept[0]["timestamp"],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Dropped {len(dropped)} pre-discussion utterances (night phase + small talk)")
    print(f"Kept {len(kept)} discussion-phase utterances, starting at "
          f"Rec_Id {kept[0]['Rec_Id']} ({kept[0]['timestamp']})")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
