"""
Build a self-contained HTML page that plays game3.mp4 alongside the
extracted audio features (ground-truth text, Whisper transcript, and
audEERING arousal/valence/dominance), synced to video playback time, for
manually spot-checking extraction quality.
"""
import base64
import json
import os

RAW_DIR = os.path.dirname(__file__) + "/../raw"
# Trimmed to the discussion phase only (76s-105s of the original 105s clip;
# see filter_discussion_phase.py) and re-encoded smaller (480p, crf 28) -
# the original 5MB full-game base64 embed was slow to load in the artifact.
VIDEO_PATH = os.path.join(RAW_DIR, "game3_discussion.mp4")
FEATURES_PATH = os.path.join(RAW_DIR, "game3_audio_features_discussion_only.json")
# Pre-fix version (naive window = [this utterance's timestamp, next one's
# timestamp)) - kept so the artifact can show a before/after comparison of
# the TRAILING_BUFFER_SEC windowing fix in extract_audio_features.py.
BEFORE_FEATURES_PATH = os.path.join(RAW_DIR, "game3_audio_features_before_windowfix_discussion_only.json")
# OSUM (Geng et al. 2025) transcript + categorical tone, run separately on
# Colab against the same discussion-phase utterances - already scoped to
# discussion-phase only (the notebook filters before running), no separate
# "_discussion_only" file needed.
OSUM_FEATURES_PATH = os.path.join(RAW_DIR, "game3_osum_features.json")
OSUM_BEFORE_FEATURES_PATH = os.path.join(RAW_DIR, "game3_osum_features_before_windowfix.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "audio_review.html")
OUT_PATH = os.path.abspath(OUT_PATH)

with open(VIDEO_PATH, "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode("ascii")

with open(FEATURES_PATH, encoding="utf-8") as f:
    features = json.load(f)

with open(BEFORE_FEATURES_PATH, encoding="utf-8") as f:
    before_features = json.load(f)
before_by_id = {u["Rec_Id"]: u["asr_transcript"] for u in before_features["utterances"]}
for u in features["utterances"]:
    u["asr_transcript_before"] = before_by_id.get(u["Rec_Id"], "")

with open(OSUM_FEATURES_PATH, encoding="utf-8") as f:
    osum_features = json.load(f)
with open(OSUM_BEFORE_FEATURES_PATH, encoding="utf-8") as f:
    osum_before_features = json.load(f)
osum_by_id = {u["Rec_Id"]: u for u in osum_features["utterances"]}
osum_before_by_id = {u["Rec_Id"]: u for u in osum_before_features["utterances"]}
for u in features["utterances"]:
    after = osum_by_id.get(u["Rec_Id"], {})
    before = osum_before_by_id.get(u["Rec_Id"], {})
    u["osum_transcript"] = after.get("osum_transcript", "")
    u["osum_transcript_before"] = before.get("osum_transcript", "")
    u["osum_tone"] = after.get("osum_tone", "")
    u["osum_tone_before"] = before.get("osum_tone", "")

# The trimmed video's timeline starts at 0, but window_sec values are still
# relative to the original full-game timeline (they start at 76) - shift
# them so they line up with the trimmed video's own clock.
cutoff_start = features["utterances"][0]["window_sec"][0]
for u in features["utterances"]:
    u["window_sec"] = [round(s - cutoff_start, 2) for s in u["window_sec"]]
trimmed_duration = features["utterances"][-1]["window_sec"][1]
features["game"]["duration_sec"] = trimmed_duration

data_json = json.dumps(features, ensure_ascii=False)

html = f"""<title>Werewolf Voice Check</title>
<style>
:root {{
  --bg: #12141a;
  --panel: #1a1d26;
  --panel-2: #20232e;
  --border: #2a2e3a;
  --text: #e6e8ec;
  --text-muted: #8b90a0;
  --accent: #d9a154;
  --accent-dim: #8a6a3d;
  --cool: #4fb3bf;
  --warm: #e2564f;
  --mono: 'IBM Plex Mono', ui-monospace, monospace;
  --sans: 'IBM Plex Sans', system-ui, sans-serif;
}}
@media (prefers-color-scheme: light) {{
  :root:not([data-theme="light"]) {{
    --bg: #f4f1ea; --panel: #ffffff; --panel-2: #faf8f3; --border: #ddd6c8;
    --text: #22242b; --text-muted: #6b6e78; --accent: #b3742a; --accent-dim: #d9a154;
  }}
}}
:root[data-theme="light"] {{
  --bg: #f4f1ea; --panel: #ffffff; --panel-2: #faf8f3; --border: #ddd6c8;
  --text: #22242b; --text-muted: #6b6e78; --accent: #b3742a; --accent-dim: #d9a154;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--text);
  font-family: var(--sans); line-height: 1.5;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 60px; }}
header {{ margin-bottom: 20px; }}
h1 {{
  font-size: 1.5rem; font-weight: 600; margin: 0 0 4px; text-wrap: balance;
  letter-spacing: -0.01em;
}}
.subtitle {{ color: var(--text-muted); font-size: 0.92rem; }}
.subtitle code {{ font-family: var(--mono); background: var(--panel-2); padding: 1px 5px; border-radius: 4px; }}

.layout {{ display: grid; grid-template-columns: 1.4fr 1fr; gap: 20px; align-items: start; }}
@media (max-width: 860px) {{ .layout {{ grid-template-columns: 1fr; }} }}

.video-panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
video {{ width: 100%; display: block; background: #000; }}

.timeline-box {{ padding: 14px 16px 16px; }}
.timeline-label {{
  display: flex; justify-content: space-between; font-family: var(--mono);
  font-size: 0.72rem; color: var(--text-muted); margin-bottom: 6px;
  text-transform: uppercase; letter-spacing: 0.06em;
}}
.timeline {{
  position: relative; height: 46px; border-radius: 8px; cursor: pointer;
  background: var(--panel-2); border: 1px solid var(--border); overflow: hidden;
}}
.tl-seg {{ position: absolute; top: 0; bottom: 0; opacity: 0.85; }}
.tl-tick {{
  position: absolute; top: 0; bottom: 0; width: 2px; background: rgba(0,0,0,0.35);
}}
.tl-playhead {{
  position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--text);
  box-shadow: 0 0 6px var(--text); pointer-events: none;
}}
.tl-legend {{ display: flex; gap: 14px; margin-top: 8px; font-size: 0.72rem; color: var(--text-muted); align-items: center; }}
.tl-swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }}

.card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; }}
.card + .card {{ margin-top: 16px; }}
.card-label {{ font-family: var(--mono); font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }}

.speaker-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
.speaker-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
.speaker-name {{ font-weight: 600; }}
.timestamp {{ font-family: var(--mono); color: var(--text-muted); font-size: 0.85rem; margin-left: auto; }}

.text-block {{ margin-bottom: 14px; }}
.text-block .k {{ font-family: var(--mono); font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 3px; }}
.text-block .v {{ font-size: 1.02rem; }}
.gt .v {{ color: var(--text); }}
.asr .v {{ color: var(--text-muted); font-style: italic; }}
.asr-compare {{ display: grid; gap: 6px; }}
.asr-compare .row {{ display: grid; grid-template-columns: 52px 1fr; gap: 8px; align-items: baseline; }}
.asr-compare .tag {{ font-family: var(--mono); font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.04em; padding: 1px 0; }}
.asr-compare .before .tag {{ color: var(--warm); }}
.asr-compare .after .tag {{ color: var(--cool); }}
.asr-compare .before .v {{ color: var(--text-muted); font-style: italic; text-decoration: line-through; text-decoration-color: var(--border); }}
.asr-compare .after .v {{ color: var(--text); font-style: italic; }}
.asr-compare .same {{ color: var(--text-muted); font-size: 0.8rem; font-style: italic; }}

.strategy-tags {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }}
.tag {{ font-family: var(--mono); font-size: 0.68rem; padding: 2px 8px; border-radius: 20px; border: 1px solid var(--accent-dim); color: var(--accent); }}

.tone-compare {{ display: flex; gap: 8px; align-items: center; margin-top: 4px; }}
.tone-pill {{ font-family: var(--mono); font-size: 0.68rem; padding: 2px 9px; border-radius: 20px; border: 1px solid currentColor; }}
.tone-arrow {{ color: var(--text-muted); font-size: 0.8rem; }}

.vad {{ display: grid; gap: 10px; margin-top: 4px; }}
.vad-row {{ display: grid; grid-template-columns: 76px 1fr 42px; align-items: center; gap: 10px; }}
.vad-row .name {{ font-family: var(--mono); font-size: 0.72rem; color: var(--text-muted); }}
.vad-track {{ height: 8px; background: var(--panel-2); border-radius: 5px; overflow: hidden; border: 1px solid var(--border); }}
.vad-fill {{ height: 100%; border-radius: 5px; transition: width 0.15s ease; }}
.vad-val {{ font-family: var(--mono); font-size: 0.78rem; text-align: right; font-variant-numeric: tabular-nums; }}

.empty-state {{ color: var(--text-muted); font-size: 0.9rem; }}

footer {{ margin-top: 28px; color: var(--text-muted); font-size: 0.78rem; }}
footer a {{ color: var(--accent); }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">

<div class="wrap">
  <header>
    <h1>Werewolf Voice Check</h1>
    <div class="subtitle">
      <code>ONE NIGHT ULTIMATE WEREWOLF  Retro 3 / Game3</code> &middot; discussion phase only (19 of 35 utterances &mdash; night phase + pre-discussion small talk dropped, see footer) &middot;
      ground truth vs Whisper + OSUM transcripts (before/after the window-timing fix), OSUM's categorical tone, and audEERING's continuous arousal / valence / dominance, synced to playback
    </div>
  </header>

  <div class="layout">
    <div>
      <div class="video-panel">
        <video id="vid" controls>
          <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div class="timeline-box">
          <div class="timeline-label"><span>arousal timeline</span><span id="tl-time">00:00</span></div>
          <div class="timeline" id="timeline"></div>
          <div class="tl-legend">
            <span><span class="tl-swatch" style="background:var(--cool)"></span>calm</span>
            <span><span class="tl-swatch" style="background:var(--warm)"></span>excited</span>
          </div>
        </div>
      </div>
    </div>

    <div>
      <div class="card" id="current-card">
        <div class="card-label">Current utterance</div>
        <div class="empty-state">Press play — the current line, transcript comparison, and vocal-emotion readout will update here as the video plays.</div>
      </div>
    </div>
  </div>

  <footer>
    Video trimmed to the discussion phase (original clip's 00:76&ndash;01:45) &mdash; night phase and pre-discussion small talk dropped, both because the app's spoken night-phase instructions bleed into the same audio track as the players' voices, and because that portion isn't the phenomenon of interest for persuasion-strategy analysis.<br>
    "Before/after" compares the original per-utterance audio window (cut exactly at the next utterance's timestamp) against a fix that extends each window by a 0.6s trailing buffer &mdash; ground-truth timestamps mark when a line was logged, not necessarily exact speech onset, so several utterances' words were being clipped into the wrong neighboring segment.<br>
    Data: <a href="https://github.com/praeploykiat/Multimind/tree/werewolf-among-us">praeploykiat/Multimind (werewolf-among-us branch)</a> &middot;
    Whisper (Radford et al., ICML 2023) &middot; audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (Wagner et al., IEEE TPAMI 2023) &middot; OSUM (Geng et al., 2025, arXiv:2501.13306)
  </footer>
</div>

<script>
const DATA = {data_json};
const utterances = DATA.utterances.filter(u => u.window_sec);
const duration = DATA.game.duration_sec;

const speakerColors = {{}};
const palette = ['#e2564f', '#4fb3bf', '#d9a154', '#8a7fd1', '#6fbf73', '#e08fc0'];
let colorIdx = 0;
for (const u of utterances) {{
  if (!(u.speaker in speakerColors)) {{
    speakerColors[u.speaker] = palette[colorIdx % palette.length];
    colorIdx++;
  }}
}}

const toneColors = {{
  happy: '#d9a154', sad: '#4fb3bf', anger: '#e2564f', neutral: '#8b90a0',
  surprise: '#8a7fd1', fear: '#6a7fd1', disgust: '#6fbf73', other: '#8b90a0',
}};

function arousalColor(a) {{
  // interpolate cool (#4fb3bf) -> warm (#e2564f) by arousal 0..1
  const cool = [79, 179, 191], warm = [226, 86, 79];
  const t = Math.max(0, Math.min(1, a));
  const rgb = cool.map((c, i) => Math.round(c + (warm[i] - c) * t));
  return `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
}}

const timeline = document.getElementById('timeline');
for (const u of utterances) {{
  const [start, end] = u.window_sec;
  const seg = document.createElement('div');
  seg.className = 'tl-seg';
  seg.style.left = (start / duration * 100) + '%';
  seg.style.width = Math.max(0.3, (end - start) / duration * 100) + '%';
  seg.style.background = u.arousal != null ? arousalColor(u.arousal) : '#444';
  seg.title = `${{u.timestamp}} ${{u.speaker}}: ${{u.utterance_ground_truth}}`;
  timeline.appendChild(seg);

  const tick = document.createElement('div');
  tick.className = 'tl-tick';
  tick.style.left = (start / duration * 100) + '%';
  timeline.appendChild(tick);
}}
const playhead = document.createElement('div');
playhead.className = 'tl-playhead';
timeline.appendChild(playhead);

const vid = document.getElementById('vid');
const card = document.getElementById('current-card');
const tlTime = document.getElementById('tl-time');

timeline.addEventListener('click', (e) => {{
  const rect = timeline.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  vid.currentTime = frac * duration;
}});

function fmtTime(t) {{
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
}}

function findCurrent(t) {{
  for (const u of utterances) {{
    const [start, end] = u.window_sec;
    if (t >= start && t < end) return u;
  }}
  return null;
}}

let lastRecId = null;
function render(t) {{
  playhead.style.left = (Math.min(t, duration) / duration * 100) + '%';
  tlTime.textContent = fmtTime(t);

  const u = findCurrent(t);
  if (!u) {{
    if (lastRecId !== null) {{
      card.innerHTML = '<div class="card-label">Current utterance</div><div class="empty-state">(silence / no annotated line at this point)</div>';
      lastRecId = null;
    }}
    return;
  }}
  if (u.Rec_Id === lastRecId) return;
  lastRecId = u.Rec_Id;

  const color = speakerColors[u.speaker];
  const tags = (u.annotation || []).map(a => `<span class="tag">${{a}}</span>`).join('');
  const vadRow = (name, val, hue) => {{
    const pct = Math.max(0, Math.min(1, val ?? 0)) * 100;
    return `<div class="vad-row">
      <div class="name">${{name}}</div>
      <div class="vad-track"><div class="vad-fill" style="width:${{pct}}%;background:${{hue}}"></div></div>
      <div class="vad-val">${{val != null ? val.toFixed(2) : '—'}}</div>
    </div>`;
  }};

  card.innerHTML = `
    <div class="card-label">Current utterance &middot; ${{u.timestamp}}</div>
    <div class="speaker-row">
      <span class="speaker-dot" style="background:${{color}}"></span>
      <span class="speaker-name">${{u.speaker}}</span>
      <span class="timestamp">Rec_Id ${{u.Rec_Id}}</span>
    </div>
    <div class="text-block gt">
      <div class="k">Ground truth</div>
      <div class="v">${{u.utterance_ground_truth || '—'}}</div>
    </div>
    <div class="text-block">
      <div class="k">Whisper transcript &mdash; before/after the window-timing fix</div>
      ${{u.asr_transcript_before === u.asr_transcript
        ? `<div class="same">(unchanged) ${{u.asr_transcript || '(empty)'}}</div>`
        : `<div class="asr-compare">
             <div class="row before"><span class="tag">before</span><span class="v">${{u.asr_transcript_before || '(empty)'}}</span></div>
             <div class="row after"><span class="tag">after</span><span class="v">${{u.asr_transcript || '(empty)'}}</span></div>
           </div>`}}
    </div>
    <div class="strategy-tags">${{tags}}</div>

    <div class="text-block" style="margin-top:16px">
      <div class="k">OSUM transcript &mdash; before/after the window-timing fix</div>
      ${{u.osum_transcript_before === u.osum_transcript
        ? `<div class="same">(unchanged) ${{u.osum_transcript || '(empty)'}}</div>`
        : `<div class="asr-compare">
             <div class="row before"><span class="tag">before</span><span class="v">${{u.osum_transcript_before || '(empty)'}}</span></div>
             <div class="row after"><span class="tag">after</span><span class="v">${{u.osum_transcript || '(empty)'}}</span></div>
           </div>`}}
    </div>
    <div class="text-block">
      <div class="k">OSUM tone (categorical)</div>
      <div class="tone-compare">
        ${{u.osum_tone_before === u.osum_tone
          ? `<span class="tone-pill" style="color:${{toneColors[u.osum_tone] || '#8b90a0'}}">${{u.osum_tone || '—'}}</span>`
          : `<span class="tone-pill" style="color:${{toneColors[u.osum_tone_before] || '#8b90a0'}}">${{u.osum_tone_before || '—'}}</span>
             <span class="tone-arrow">&rarr;</span>
             <span class="tone-pill" style="color:${{toneColors[u.osum_tone] || '#8b90a0'}}">${{u.osum_tone || '—'}}</span>`}}
      </div>
    </div>

    <div class="card-label" style="margin-top:16px">Vocal emotion, continuous (audEERING)</div>
    <div class="vad">
      ${{vadRow('Arousal', u.arousal, 'var(--warm)')}}
      ${{vadRow('Valence', u.valence, 'var(--cool)')}}
      ${{vadRow('Dominance', u.dominance, 'var(--accent)')}}
    </div>
  `;
}}

vid.addEventListener('timeupdate', () => render(vid.currentTime));
vid.addEventListener('seeking', () => render(vid.currentTime));
render(0);
</script>
"""

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Written to {OUT_PATH}, size = {os.path.getsize(OUT_PATH) / 1e6:.2f} MB")
