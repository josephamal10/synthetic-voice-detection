import datetime
import html
import io
import itertools
import os
import tempfile
import uuid

import auth
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from streamlit_mic_recorder import mic_recorder
from streamlit_option_menu import option_menu

# ============================================================
# CONFIG — must match training exactly
# ============================================================
MFCC_MEAN = -3.1552699
MFCC_STD = 29.653503
N_MFCC = 40
MAX_LEN = 400
SAMPLE_RATE = 16000
MIN_DURATION_SEC = 0.3
DEFAULT_THRESHOLD = 0.5
PEAK_NORM_TARGET = 0.95
SILENCE_PEAK_THRESHOLD = 0.003

BASELINE_MODEL = "best_model.pt"
FINETUNED_MODEL = "best_model_finetuned.pt"
COMBINED_MODEL = "best_model_combined.pt"
COMBINED_V2_MODEL = "best_model_v2.pt"

# Measured results. ASVspoof dev = 24,844 studio clips; In-the-Wild val = 6,355 real-world clips.
# (accuracy, bona-fide recall, bona-fide precision); "eer" = (ASVspoof EER, In-the-Wild EER),
# None where not yet measured — display code must handle that, not assume it's always present.
MODEL_SCORES = {
    BASELINE_MODEL: {"asv": (0.976, 0.847, 0.913), "itw": (0.468, 0.230, 0.749),
                     "eer": (0.0583, 0.4152)},
    FINETUNED_MODEL: {"asv": (0.760, 0.960, 0.295), "itw": (0.956, 0.968, 0.963),
                      "eer": (0.0757, 0.0445)},
    COMBINED_MODEL: {"asv": (0.974, 0.779, 0.963), "itw": (0.965, 0.984, 0.962),
                     "eer": (0.0847, 0.0356)},
    COMBINED_V2_MODEL: {"asv": (0.976, 0.815, 0.942), "itw": (0.961, 0.957, 0.981),
                        "eer": (0.0833, 0.0360)},
}

# Score at which false acceptances and false rejections are equal, on ASVspoof dev.
# No entry for a model means EER hasn't been measured for it yet.
EER_THRESHOLDS = {
    BASELINE_MODEL: 0.0074,
    FINETUNED_MODEL: 0.8572,
    COMBINED_MODEL: 0.0042,
    COMBINED_V2_MODEL: 0.0095,
}

MODEL_NOTES = {
    BASELINE_MODEL: "Trained on ASVspoof 2019 LA only. Accurate on studio audio, but "
                    "misclassifies most real-world genuine voices as synthetic.",
    FINETUNED_MODEL: "Adapted to real-world audio, but suffered catastrophic forgetting — "
                     "it now lets most ASVspoof-style spoofs through as genuine.",
    COMBINED_MODEL: "Trained on ASVspoof and In-the-Wild together. Retains studio accuracy "
                    "while handling real-world recordings, but still misses synthesis methods "
                    "absent from both datasets (e.g. classical formant-based TTS).",
    COMBINED_V2_MODEL: "Trained on ASVspoof, In-the-Wild, and a small classical-TTS sample set "
                       "(Windows SAPI, espeak-ng) added to close that specific gap. Matches or "
                       "slightly improves on the previous model everywhere it was measured. "
                       "Recommended.",
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Timestamps are recorded in Indian Standard Time, not the server's clock —
# cloud hosts run on UTC, which would show every analysis 5.5 hours early.
# A fixed offset (IST has no daylight saving) needs no tzdata package.
APP_TZ = datetime.timezone(datetime.timedelta(hours=5, minutes=30), "IST")


def now_local():
    return datetime.datetime.now(APP_TZ)

# Fresh integer per gauge/sparkline render, so each gets its own uniquely
# named CSS keyframes/classes — several can be on screen at once (a
# multi-file upload analyses several clips in the same rerun) and must not
# collide and animate each other to the wrong value.
_uid_counter = itertools.count()

# "Cyan-Slate Lab" — a cool, deep teal-cyan instrument palette. Picked over a
# blue-to-cyan-gradient "AI product" look and over the original cobalt/
# periwinkle palette after a round of side-by-side comparisons; reads as a
# clean clinical lab/security tool rather than a generic AI demo.
C_DARK = {
    "bg": "#0a1418",
    "panel": "#0f1e24",
    "panel2": "#142832",
    "border": "#1c333c",
    "text": "#eef6f8",
    "muted": "#82a3ac",
    "accent": "#14b8a6",
    "accent2": "#5eead4",
    "genuine": "#1fae7a",
    "synthetic": "#e2584f",
    "warn": "#d99a3d",
}

# Same instrument, same spirit, inverted to a bright ground — crisp
# white-cyan panels, deep teal-black ink text. Accent/genuine/synthetic/warn
# are all deepened a notch from their dark-mode values for contrast against
# white rather than near-black.
C_LIGHT = {
    "bg": "#f2f9fa",
    "panel": "#ffffff",
    "panel2": "#e3f2f1",
    "border": "#cfe3e1",
    "text": "#0d2b2b",
    "muted": "#5b7d7a",
    "accent": "#0f8a7a",
    "accent2": "#0d9488",
    "genuine": "#178f63",
    "synthetic": "#c23f37",
    "warn": "#a8672a",
}

# The active palette every function below reads via C['key'] — swapped in
# place (not reassigned) at the top of each rerun based on the sidebar's
# dark/light toggle, so every function that already does C['...'] picks up
# the current theme automatically with no other code needing to change.
C = dict(C_DARK)


# ============================================================
# MODEL — must match training exactly
# ============================================================
class VoiceDetectionModel(nn.Module):
    def __init__(self):
        super(VoiceDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=32 * 10, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(64 * 2, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        batch_size, channels, freq, time_dim = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time_dim, channels * freq)
        lstm_out, _ = self.lstm(x)
        x = lstm_out.mean(dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@st.cache_resource
def load_model(path):
    model = VoiceDetectionModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def available_models():
    """Ordered worst→best so the last entry is the recommended default."""
    candidates = [
        ("1 · Baseline (ASVspoof only)", BASELINE_MODEL),
        ("2 · Fine-tuned (In-the-Wild only)", FINETUNED_MODEL),
        ("3 · Combined (ASVspoof + In-the-Wild)", COMBINED_MODEL),
        ("4 · Combined + classical TTS (best)", COMBINED_V2_MODEL),
    ]
    return {label: path for label, path in candidates if os.path.exists(path)}


def extract_mfcc(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, max_len=MAX_LEN):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc


def predict(model, y, sr, threshold=DEFAULT_THRESHOLD):
    """Returns (label, margin, p_genuine, mfcc).

    p_genuine is the raw model score. `margin` expresses how far that score sits
    from the decision threshold, scaled to 0-1 within whichever side it fell on,
    so it stays meaningful when the threshold is moved away from 0.5.
    """
    mfcc = extract_mfcc(y, sr=sr)
    mfcc_norm = (mfcc - MFCC_MEAN) / MFCC_STD
    x = torch.tensor(mfcc_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        p_genuine = torch.sigmoid(model(x)).item()
    if p_genuine > threshold:
        label = "Genuine"
        margin = (p_genuine - threshold) / max(1.0 - threshold, 1e-9)
    else:
        label = "Synthetic"
        margin = (threshold - p_genuine) / max(threshold, 1e-9)
    return label, min(max(margin, 0.0), 1.0), p_genuine, mfcc


def load_audio_safely(raw_bytes, sr=SAMPLE_RATE):
    """Returns (y, sr, error_message); error_message is None on success.

    Loads via a temporary file rather than an in-memory buffer. Browser
    microphone recordings usually arrive as WebM/Opus even when labelled
    ".wav", which soundfile/libsndfile cannot read on its own — librosa's
    ffmpeg-backed fallback only activates for a real file path, never for
    a BytesIO object, so an in-memory buffer would silently skip it.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        y, sr = librosa.load(tmp_path, sr=sr, mono=True)
    except Exception as exc:
        return None, None, f"Could not decode this audio ({exc.__class__.__name__}). Try another file or format."
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    if y is None or len(y) == 0:
        return None, None, "This file contains no audio samples."
    if len(y) / sr < MIN_DURATION_SEC:
        return None, None, f"Clip is too short to analyse (under {MIN_DURATION_SEC}s)."
    # Peak-normalize so a quietly-captured clip (a common browser-microphone
    # issue — low input gain, distance from the mic) isn't scored on MFCCs an
    # order of magnitude smaller than anything the model saw in training,
    # which reads to it as out-of-distribution and gets flagged Synthetic.
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak <= SILENCE_PEAK_THRESHOLD:
        # Genuinely no signal — usually a muted/wrong microphone or a denied
        # browser mic permission, not a quiet voice. Scaling this up would
        # just blast noise floor to full volume, and feeding it to the model
        # anyway produces a meaningless but confident-looking verdict (seen
        # in practice: reads as "100% Synthetic"). Surface the real problem
        # instead, with the measured peak so it's diagnosable.
        return None, None, (
            f"This recording is essentially silent (peak level {peak:.6f}). Check that the "
            "correct microphone is selected and unmuted in your OS sound settings, and that "
            "this site has microphone permission in the browser, then try again."
        )
    y = y / peak * PEAK_NORM_TARGET
    return y, sr, None


# ============================================================
# UI HELPERS
# ============================================================
def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');

        html, body, .stApp, [class*="st-emotion"] {{
            font-family: 'Space Grotesk', -apple-system, sans-serif;
        }}
        /* Streamlit's Material icons are ligature fonts — the rule above would
           otherwise render them as their literal names ("upload", "arrow_right"). */
        [data-testid="stIconMaterial"], [data-testid^="stExpanderIcon"],
        .material-symbols-rounded, .material-icons, [class*="material-symbols"] {{
            font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        }}
        /* A very slow drift on where the corner glow sits — 28s, meant to be
           almost impossible to catch in the act, so the background reads as
           alive rather than static without ever competing for attention. */
        @property --dx {{ syntax: '<percentage>'; inherits: true; initial-value: 12%; }}
        @property --dy {{ syntax: '<percentage>'; inherits: true; initial-value: -8%; }}
        @keyframes bgDrift {{
            0%   {{ --dx: 12%; --dy: -8%; }}
            50%  {{ --dx: 82%; --dy: 10%; }}
            100% {{ --dx: 12%; --dy: -8%; }}
        }}
        .stApp {{
            background:
                radial-gradient(1000px 520px at var(--dx, 12%) var(--dy, -8%), {C['accent']}12 0%, transparent 58%),
                radial-gradient(760px 420px at 92% -4%, {C['accent2']}0e 0%, transparent 52%),
                repeating-linear-gradient(0deg, {C['accent']}05 0 1px, transparent 1px 44px),
                repeating-linear-gradient(90deg, {C['accent']}05 0 1px, transparent 1px 44px),
                {C['bg']};
            color: {C['text']};
            animation: bgDrift 28s ease-in-out infinite;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {C['panel2']} 0%, {C['bg']} 100%);
            border-right: 1px solid {C['border']};
        }}
        #MainMenu, footer {{ visibility: hidden; }}

        /* ---------- hero with animated equaliser ---------- */
        .hero {{
            padding: 34px 36px 30px 36px;
            border-radius: 18px;
            background: linear-gradient(135deg, {C['panel2']} 0%, {C['panel']} 100%);
            border: 1px solid {C['border']};
            margin-bottom: 10px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 0 0 1px {C['accent']}0d, 0 18px 48px -22px {C['accent']}40;
        }}
        .hero:before {{
            content: "";
            position: absolute; inset: 0;
            background: radial-gradient(560px 240px at 88% 12%, {C['accent']}1a, transparent 70%);
            pointer-events: none;
        }}
        .hero:after {{
            content: "";
            position: absolute; left: 0; right: 0; top: 0; height: 2px;
            background: linear-gradient(90deg, transparent, {C['accent']}, {C['accent2']}, transparent);
            animation: sweep 5s ease-in-out infinite;
        }}
        @keyframes sweep {{
            0%, 100% {{ opacity: .25; transform: translateX(-18%); }}
            50%      {{ opacity: 1;   transform: translateX(18%); }}
        }}
        .hero h1 {{
            margin: 0; font-size: 36px; font-weight: 700; letter-spacing: -0.8px;
            background: linear-gradient(92deg, {C['accent']} 10%, {C['text']} 55%, {C['accent2']} 95%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            position: relative; z-index: 1;
        }}
        /* Sidebar logo — same gradient-text-clip trick as .hero h1 above.
           Must live in this stylesheet, not an inline style="" attribute:
           Streamlit's HTML sanitizer silently drops -webkit-background-clip
           from inline styles (it kept -webkit-text-fill-color, oddly), which
           left the logo rendering as a solid filled block with no visible
           letters at all. */
        .brand-title {{
            font-size: 21px; font-weight: 700; letter-spacing: -.4px;
            background: linear-gradient(90deg, {C['accent']}, {C['accent2']});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .hero p {{
            color: {C['muted']}; margin-top: 10px; font-size: 15px; max-width: 660px;
            line-height: 1.6; position: relative; z-index: 1;
        }}
        .eq {{ position: absolute; right: 34px; bottom: 26px; display: flex;
               align-items: flex-end; gap: 4px; height: 46px; opacity: .85; }}
        .eq i {{
            display: block; width: 4px; border-radius: 2px;
            background: linear-gradient(180deg, {C['accent']}, {C['accent']}30);
            animation: bounce 1.15s ease-in-out infinite;
        }}
        @keyframes bounce {{
            0%, 100% {{ height: 8px;  opacity: .45; }}
            50%      {{ height: 42px; opacity: 1; }}
        }}

        .pill {{
            display: inline-block; padding: 5px 13px; border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10.5px; font-weight: 700; letter-spacing: 1.4px; text-transform: uppercase;
            background: {C['accent']}14; color: {C['accent']};
            border: 1px solid {C['accent']}3a; margin-bottom: 16px;
            position: relative; z-index: 1;
        }}
        /* A functional "you are here" signifier next to the pill — not
           decoration. (The sidebar's own current-page highlight lives inside
           a third-party component's iframe, which this app's CSS can't
           reach, so the signal lives here instead.) */
        .live-dot {{
            display: inline-block; width: 7px; height: 7px; border-radius: 50%;
            background: {C['genuine']}; margin-left: 9px; margin-bottom: 16px;
            vertical-align: middle; position: relative; z-index: 1;
            animation: liveBreathe 2.6s ease-in-out infinite;
        }}
        @keyframes liveBreathe {{
            0%, 100% {{ opacity: .5; box-shadow: 0 0 0 0 {C['genuine']}00; }}
            50%      {{ opacity: 1;  box-shadow: 0 0 9px 2px {C['genuine']}80; }}
        }}

        /* ---------- cards ---------- */
        .card {{
            background: {C['panel']};
            border: 1px solid {C['border']};
            border-radius: 14px; padding: 20px 22px; height: 100%;
            position: relative; overflow: hidden;
            transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
        }}
        .card:before {{
            content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 2px;
            background: linear-gradient(180deg, {C['accent']}, transparent);
            opacity: .6;
        }}
        /* Cursor-tracked spotlight — the JS (inject_micro_interactions) only
           ever sets --mx/--my and toggles this class; the glow itself, and
           its fade in/out, are plain CSS. */
        .card:after {{
            content: ""; position: absolute; inset: -1px; pointer-events: none; opacity: 0;
            background: radial-gradient(240px circle at var(--mx, 50%) var(--my, 50%),
                        {C['accent']}30, transparent 62%);
            transition: opacity .25s ease;
        }}
        .card.spotlit:after {{ opacity: 1; }}
        .card:hover {{
            border-color: {C['accent']}44;
            box-shadow: 0 14px 34px -18px {C['accent']}66;
        }}
        .card h4 {{ margin: 0 0 6px 0; font-size: 15px; color: {C['text']}; font-weight: 700; }}
        .card p {{ margin: 0; font-size: 13px; color: {C['muted']}; line-height: 1.6; }}
        .card .ico {{ font-size: 22px; display: block; margin-bottom: 12px; }}

        /* ---------- stats ---------- */
        .stat {{
            background: linear-gradient(160deg, {C['panel2']}, {C['panel']});
            border: 1px solid {C['border']};
            border-radius: 14px; padding: 18px 14px; text-align: center;
            position: relative; overflow: hidden;
            min-height: 104px;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            transition: border-color .18s ease, box-shadow .18s ease;
        }}
        .stat:hover {{ border-color: {C['accent']}40; box-shadow: 0 0 26px -12px {C['accent']}80; }}
        .stat .v {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 24px; font-weight: 700; color: {C['accent']};
            text-shadow: 0 0 20px {C['accent']}55; line-height: 1.2;
        }}
        .stat .k {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 9.5px; color: {C['muted']}; text-transform: uppercase;
            letter-spacing: 1.3px; margin-top: 7px;
        }}

        /* ---------- numbered steps ---------- */
        .step {{
            display: flex; gap: 16px; align-items: flex-start;
            background: {C['panel']}; border: 1px solid {C['border']};
            border-radius: 12px; padding: 16px 18px; margin-bottom: 10px;
            transition: border-color .18s ease, transform .18s ease;
        }}
        .step:hover {{ border-color: {C['accent']}3a; transform: translateX(3px); }}
        .step .n {{
            flex: 0 0 32px; height: 32px; border-radius: 8px;
            background: linear-gradient(135deg, {C['accent']}, {C['accent2']});
            color: {C['bg']}; font-family: 'JetBrains Mono', monospace;
            font-weight: 700; display: flex;
            align-items: center; justify-content: center; font-size: 14px;
            box-shadow: 0 0 18px -4px {C['accent']}90;
        }}
        .step h5 {{ margin: 3px 0 5px 0; font-size: 14px; color: {C['text']}; font-weight: 700; }}
        .step p {{ margin: 0; font-size: 13px; color: {C['muted']}; line-height: 1.6; }}

        /* ---------- banner ---------- */
        .banner {{
            border-radius: 12px; padding: 15px 20px; margin: 8px 0 14px 0;
            border-left: 3px solid {C['warn']};
            background: linear-gradient(90deg, {C['warn']}12, {C['warn']}04);
            color: {C['text']}; font-size: 13.5px; line-height: 1.6;
        }}
        .banner b {{ color: {C['warn']}; }}

        /* ---------- session report: recordings library + case file ---------- */
        [class*="st-key-clipcard_"] {{
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }}
        [class*="st-key-clipcard_"]:hover {{ transform: translateY(-3px); }}
        [class*="st-key-clipcard_gen_"] {{ border-top: 3px solid {C['genuine']} !important; }}
        [class*="st-key-clipcard_syn_"] {{ border-top: 3px solid {C['synthetic']} !important; }}
        [class*="st-key-clipcard_gen_"]:hover {{
            border-color: {C['genuine']}77 !important; box-shadow: 0 18px 40px -22px {C['genuine']}bb !important;
        }}
        [class*="st-key-clipcard_syn_"]:hover {{
            border-color: {C['synthetic']}77 !important; box-shadow: 0 18px 40px -22px {C['synthetic']}bb !important;
        }}
        .clip-top {{ display: flex; justify-content: space-between; align-items: center; }}
        .clip-chip {{
            display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 10px;
            font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase;
            padding: 4px 10px; border-radius: 999px; border: 1px solid;
        }}
        .clip-no {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: {C['muted']}; }}
        .clip-name {{
            margin: 12px 0 10px 0; font-weight: 700; font-size: 14.5px; color: {C['text']};
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }}
        .clip-wave {{
            background: {C['bg']}88; border: 1px solid {C['border']}; border-radius: 10px; padding: 8px 10px;
        }}
        .clip-nowave {{
            height: 44px; display: flex; align-items: center; justify-content: center;
            color: {C['muted']}; font-size: 11px;
        }}
        .clip-score {{
            display: flex; align-items: center; gap: 10px; margin-top: 12px;
            font-size: 11px; color: {C['muted']};
        }}
        .clip-score .bar {{ flex: 1; height: 5px; border-radius: 3px; background: {C['border']}; overflow: hidden; }}
        .clip-score .bar i {{ display: block; height: 100%; border-radius: 3px; }}
        .clip-score b {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; }}
        .clip-meta {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0 4px 0; }}
        .clip-meta span {{
            font-size: 11px; color: {C['muted']}; background: {C['panel2']};
            border: 1px solid {C['border']}; border-radius: 6px; padding: 3px 8px;
        }}
        .split-bar {{
            display: flex; height: 6px; border-radius: 3px; overflow: hidden; gap: 2px;
            margin: 14px 0 6px 0; background: {C['border']};
        }}
        .split-bar i {{ display: block; height: 100%; }}
        .case-head {{
            position: relative; overflow: hidden; border: 1px solid; border-radius: 20px;
            padding: 28px 30px; margin: 14px 0 12px 0;
            display: flex; justify-content: space-between; align-items: center; gap: 24px; flex-wrap: wrap;
            animation: riseIn .5s ease both;
        }}
        .case-backdrop {{
            position: absolute; left: 0; right: 0; bottom: -8px; opacity: .16; pointer-events: none;
        }}
        .case-main, .case-score {{ position: relative; z-index: 1; min-width: 0; }}
        .case-main {{ flex: 1 1 360px; }}
        .case-kicker {{
            font-family: 'JetBrains Mono', monospace; font-size: 10.5px; letter-spacing: 1.4px;
            text-transform: uppercase; color: {C['muted']};
        }}
        .case-title {{
            font-size: 28px; font-weight: 800; color: {C['text']}; margin: 8px 0 14px 0;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }}
        .case-score {{ text-align: right; }}
        .case-score .v {{
            font-family: 'JetBrains Mono', monospace; font-size: 46px; font-weight: 800; line-height: 1;
        }}
        .case-score .k {{
            font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 1.3px;
            text-transform: uppercase; color: {C['muted']}; margin-top: 6px;
        }}
        .case-conf {{
            display: inline-block; margin-top: 10px; padding: 4px 11px; border: 1px solid;
            border-radius: 999px; font-size: 11.5px; color: {C['text']}; background: {C['panel']}aa;
        }}
        .case-facts {{
            display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-bottom: 8px;
        }}
        .case-facts div {{
            background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 12px; padding: 11px 14px;
        }}
        .case-facts span {{
            display: block; font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
            letter-spacing: 1.2px; text-transform: uppercase; color: {C['muted']};
        }}
        .case-facts b {{
            display: block; margin-top: 4px; font-size: 13.5px; color: {C['text']};
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }}
        .case-wave {{ padding: 6px 4px 10px 4px; }}
        @media (max-width: 900px) {{
            .case-facts {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            .case-score {{ text-align: left; }}
        }}

        /* ---------- streamlit widgets ---------- */
        div[data-testid="stFileUploaderDropzone"] {{
            background: linear-gradient(160deg, {C['panel2']}, {C['panel']});
            border: 1.5px dashed {C['accent']}55;
            border-radius: 16px; padding: 6px;
            transition: border-color .2s ease, box-shadow .2s ease;
        }}
        div[data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {C['accent']};
            box-shadow: 0 0 30px -10px {C['accent']}70;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {C['muted']}; font-size: 13.5px; letter-spacing: .2px;
        }}
        .stTabs [aria-selected="true"] {{ color: {C['accent']} !important; }}
        .stTabs [data-baseweb="tab-highlight"] {{ background: {C['accent']} !important; }}

        /* ---------- bordered containers (input console panels) ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid {C['border']} !important;
            border-radius: 18px !important;
            background: linear-gradient(165deg, {C['panel2']}cc, {C['panel']}cc);
            box-shadow: 0 0 0 1px {C['accent']}0a, 0 20px 48px -28px {C['accent']}55;
        }}

        /* ---------- microphone orb ---------- */
        .mic-orb {{
            width: 78px; height: 78px; border-radius: 50%; margin: 4px auto 0 auto;
            display: flex; align-items: center; justify-content: center; font-size: 32px;
            background: radial-gradient(circle at 34% 30%, {C['accent2']}, {C['accent']} 72%);
            animation: micpulse 2.2s ease-out infinite;
        }}
        @keyframes micpulse {{
            0%   {{ box-shadow: 0 0 0 0 {C['accent']}55, 0 0 0 0 {C['accent2']}33; }}
            70%  {{ box-shadow: 0 0 0 16px {C['accent']}00, 0 0 0 32px {C['accent2']}00; }}
            100% {{ box-shadow: 0 0 0 0 {C['accent']}00, 0 0 0 0 {C['accent2']}00; }}
        }}
        .upload-glyph {{
            width: 78px; height: 78px; border-radius: 22px; margin: 4px auto 0 auto;
            display: flex; align-items: center; justify-content: center; font-size: 30px;
            background: linear-gradient(155deg, {C['accent']}22, {C['accent2']}14);
            border: 1px solid {C['accent']}40;
        }}
        .io-caption {{ text-align: center; padding: 2px 0 16px 0; }}
        .io-caption .t {{
            color: {C['text']}; font-weight: 700; font-size: 15px; margin-top: 12px;
        }}
        .io-caption .s {{ color: {C['muted']}; font-size: 12.5px; margin-top: 3px; }}
        table {{ color: {C['text']} !important; font-size: 13px; border-collapse: separate !important; }}
        thead th {{
            color: {C['accent']} !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 11px !important; letter-spacing: .5px;
            background: {C['panel2']} !important;
        }}
        tbody tr:hover td {{ background: {C['accent']}0d !important; }}
        h3 {{
            font-size: 19px !important; font-weight: 700 !important;
            letter-spacing: -.3px; margin-top: 6px !important;
        }}
        h3:before {{
            content: "▸ "; color: {C['accent']}; font-size: 15px;
        }}
        code {{ color: {C['accent2']} !important; font-family: 'JetBrains Mono', monospace !important; }}

        /* ---------- buttons ---------- */
        div[data-testid="stButton"] button {{
            position: relative; overflow: hidden;
            border-radius: 10px !important; font-weight: 600 !important;
            border: 1px solid {C['border']} !important;
            transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
        }}
        div[data-testid="stButton"] button:hover {{
            transform: translateY(-1px); border-color: {C['accent']}80 !important;
            box-shadow: 0 8px 20px -10px {C['accent']}70;
        }}
        div[data-testid="stButton"] button:active {{ transform: translateY(0) scale(.97); }}
        div[data-testid="stButton"] button:focus-visible {{
            outline: 2px solid {C['accent']} !important; outline-offset: 2px;
        }}
        div[data-testid="stButton"] button[kind="primary"] {{
            background: linear-gradient(135deg, {C['accent']}, {C['accent2']}) !important;
            border: none !important; color: {C['bg']} !important;
        }}
        div[data-testid="stButton"] button[kind="primary"]:hover {{
            box-shadow: 0 10px 26px -10px {C['accent']}90;
        }}
        /* Click-position ripple — inject_micro_interactions() adds the .vg-ripple
           span at the click point on primary buttons; this is just its look. */
        .vg-ripple {{
            position: absolute; border-radius: 50%; background: rgba(255,255,255,.35);
            transform: scale(0); pointer-events: none;
            animation: vgRippleAnim .6s ease-out forwards;
        }}
        @keyframes vgRippleAnim {{ to {{ transform: scale(1); opacity: 0; }} }}

        /* ---------- skeleton shimmer (model loading) ---------- */
        .skeleton-card {{
            background: {C['panel']}; border: 1px solid {C['border']};
            border-radius: 14px; padding: 14px 18px; margin: 2px 0 12px 0;
        }}
        .skel-line, .skel-block {{
            background: linear-gradient(90deg, {C['panel2']} 25%, {C['border']} 37%, {C['panel2']} 63%);
            background-size: 400% 100%;
            animation: skelShimmer 1.6s ease-in-out infinite;
            border-radius: 4px;
        }}
        .skel-line {{ height: 12px; margin-bottom: 10px; }}
        .skel-block {{ height: 40px; }}
        @keyframes skelShimmer {{ 0% {{ background-position: 100% 0; }} 100% {{ background-position: 0 0; }} }}

        /* ---------- theme toggle switch ---------- */
        /* Streamlit's own default (light) widget theme is used for native
           controls since there's no .streamlit/config.toml — normally close
           enough against a dark custom background, but a light-on-light
           switch track became nearly invisible once the light palette made
           the page background light too. Driven by our own tokens instead,
           scoped to this one control via its aria-label (not a hashed
           st-emotion-cache class, which isn't stable across versions). */
        label:has(input[aria-label="Dark mode"]) > div:first-of-type {{
            background: {C['border']} !important; border: 1px solid {C['border']} !important;
        }}
        label:has(input[aria-label="Dark mode"]):has(input:checked) > div:first-of-type {{
            background: {C['accent']} !important; border-color: {C['accent']} !important;
        }}
        label:has(input[aria-label="Dark mode"]) > div:first-of-type > div {{
            background: {C['text']} !important;
        }}
        label:has(input[aria-label="Dark mode"]) [data-testid="stWidgetLabel"] p {{
            color: {C['text']} !important;
        }}

        /* ---------- alerts announce themselves ---------- */
        div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {{
            animation: alertShake .4s ease-out;
        }}
        @keyframes alertShake {{
            10%, 90% {{ transform: translateX(-1px); }}
            20%, 80% {{ transform: translateX(2px); }}
            30%, 50%, 70% {{ transform: translateX(-3px); }}
            40%, 60% {{ transform: translateX(3px); }}
        }}

        /* ---------- gentle entrance for each page's hero ---------- */
        @keyframes riseIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .hero {{ animation: riseIn .45s ease-out; }}

        /* ---------- scan sweep while a clip is actually being analysed ---------- */
        div[data-testid="stSpinner"] {{
            position: relative; overflow: hidden;
            background: {C['panel']}; border: 1px solid {C['accent']}40;
            border-radius: 12px; padding: 14px 18px !important;
        }}
        div[data-testid="stSpinner"]::after {{
            content: ""; position: absolute; top: 0; bottom: 0; width: 45%; left: -45%;
            background: linear-gradient(90deg, transparent, {C['accent']}30, {C['accent']}70,
                        {C['accent']}30, transparent);
            animation: scanSweep 1.3s ease-in-out infinite;
            pointer-events: none;
        }}
        @keyframes scanSweep {{
            0%   {{ left: -45%; }}
            100% {{ left: 100%; }}
        }}

        /* ---------- waveform / signal plots draw in when they appear ---------- */
        div[data-testid="stExpanderDetails"] div[data-testid="stImage"] img {{
            animation: imgReveal 1.1s cubic-bezier(.16,.84,.44,1);
        }}
        @keyframes imgReveal {{
            from {{ clip-path: inset(0 100% 0 0); }}
            to   {{ clip-path: inset(0 0 0 0); }}
        }}

        /* Reduced motion: every reveal above always sets its OWN correct
           resting value outside the animation, so switching animation off
           here can never leave a score, arc or sparkline stuck showing a
           mid-reveal (i.e. wrong) value — only the motion itself is cut. */
        @media (prefers-reduced-motion: reduce) {{
            .anim-reveal {{ animation: none !important; }}
            div[data-testid="stSpinner"]::after {{ display: none !important; }}
            div[data-testid="stExpanderDetails"] div[data-testid="stImage"] img {{ animation: none !important; }}
            .stApp {{ animation: none !important; }}
            .card {{ transform: none !important; }}
            .card:after {{ display: none !important; }}
            .vg-ripple {{ display: none !important; }}
            div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {{ animation: none !important; }}
            .live-dot {{ animation: none !important; opacity: 1; }}
            .skel-line, .skel-block {{ animation: none !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_particles(echo=False):
    """Cursor-reactive particle field painted behind the whole app.

    Runs from a zero-height component iframe, which is same-origin on localhost,
    so it can attach a canvas to the parent document. If that access is ever
    blocked the whole thing is skipped silently — it is decoration only.

    `echo=True` (login page only): dots are additionally pushed outward by an
    invisible shockwave centred on the same fixed viewport point (50% / 42%)
    as the visible `.auth-ring` CSS pulse in inject_auth_theme, using the
    identical period/delays/radius range so the two stay in lockstep — the
    dust field visibly reacts to each "echo" the same way it already reacts
    to the cursor, then springs back.
    """
    echo_js = "true" if echo else "false"
    components.html(
        f"""
        <script>
        (function () {{
          try {{
            const doc = window.parent.document;
            const app = doc.querySelector('.stApp');
            if (!app) return;

            // Streamlit reruns re-execute this: tear down the previous field first.
            const prev = doc.getElementById('vg-particles');
            if (prev) {{ clearInterval(prev.__timer); prev.remove(); }}

            const cv = doc.createElement('canvas');
            cv.id = 'vg-particles';
            cv.style.cssText =
              'position:fixed;top:0;left:0;z-index:0;pointer-events:none;';
            app.insertBefore(cv, app.firstChild);

            // Keep Streamlit's own content stacked above the canvas.
            let st = doc.getElementById('vg-particles-css');
            if (!st) {{
              st = doc.createElement('style');
              st.id = 'vg-particles-css';
              st.textContent =
                '[data-testid="stAppViewContainer"],[data-testid="stHeader"],' +
                'section[data-testid="stSidebar"]{{position:relative;z-index:1;}}' +
                '[data-testid="stAppViewContainer"]{{background:transparent!important;}}';
              doc.head.appendChild(st);
            }}

            const ctx = cv.getContext('2d');
            const COLORS = ['{C["accent"]}', '{C["accent2"]}', '{C["genuine"]}'];
            let W = 0, H = 0, dots = [];

            function build() {{
              // Size from the parent viewport: the canvas lives in a fixed-position
              // layer whose percentage sizing is not reliable at injection time.
              const dpr = window.parent.devicePixelRatio || 1;
              W = window.parent.innerWidth || 1280;
              H = window.parent.innerHeight || 720;
              cv.style.width = W + 'px';
              cv.style.height = H + 'px';
              cv.width = W * dpr; cv.height = H * dpr;
              ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
              const n = Math.min(340, Math.round((W * H) / 3400));
              dots = Array.from({{length: n}}, () => {{
                const x = Math.random() * W, y = Math.random() * H;
                return {{
                  hx: x, hy: y, x: x, y: y, vx: 0, vy: 0,
                  r: 1.1 + Math.random() * 2.4,
                  c: COLORS[(Math.random() * COLORS.length) | 0],
                  a: 0.20 + Math.random() * 0.45,
                  dz: 0.15 + Math.random() * 0.5,          // drift speed
                  ph: Math.random() * Math.PI * 2          // drift phase
                }};
              }});
            }}

            const mouse = {{x: -9999, y: -9999}};
            doc.addEventListener('mousemove', e => {{ mouse.x = e.clientX; mouse.y = e.clientY; }});
            doc.addEventListener('mouseleave', () => {{ mouse.x = -9999; mouse.y = -9999; }});
            window.parent.addEventListener('resize', build);

            const R = 180;            // cursor influence radius

            // ---- mic-echo shockwave: mirrors the .auth-ring CSS animation
            // (see inject_auth_theme) exactly, so the invisible force below
            // and the visible expanding ring the user sees stay in lockstep.
            // Same fixed viewport point the CSS ring uses (top:42%;left:50%),
            // not the mic icon's own DOM rect, since that's how the ring
            // itself is positioned — matching that guarantees they line up.
            const echoOn = {echo_js};
            const RING_PERIOD = 4500, RING_DELAYS = [0, 1500, 3000];
            const RING_R0 = 40, RING_R1 = 1500, RING_BAND = 110;
            const start = Date.now();

            let t = 0;

            function frame() {{
              t += 0.012;
              ctx.clearRect(0, 0, W, H);

              let rings = [];
              const originX = W * 0.5, originY = H * 0.42;
              if (echoOn) {{
                const now = Date.now();
                for (const delay of RING_DELAYS) {{
                  let pt = ((now - start - delay) % RING_PERIOD + RING_PERIOD) % RING_PERIOD / RING_PERIOD;
                  const radius = RING_R0 + (RING_R1 - RING_R0) * (1 - Math.pow(1 - pt, 2));
                  const strength = pt < 0.8 ? (0.9 - pt / 0.8 * 0.7) : (0.2 - (pt - 0.8) / 0.2 * 0.2);
                  rings.push({{r: radius, s: Math.max(0, strength)}});
                }}
              }}

              for (const d of dots) {{
                // gentle ambient drift around the home position
                const dx0 = Math.cos(t + d.ph) * d.dz * 6;
                const dy0 = Math.sin(t * 0.9 + d.ph) * d.dz * 6;
                const tx = d.hx + dx0, ty = d.hy + dy0;

                // repulsion from the cursor, easing off with distance
                let px = 0, py = 0;
                const mx = d.x - mouse.x, my = d.y - mouse.y;
                const dist = Math.hypot(mx, my);
                if (dist < R && dist > 0.001) {{
                  const f = (1 - dist / R);
                  px = (mx / dist) * f * 105;
                  py = (my / dist) * f * 105;
                }}

                // radial push from each passing echo ring, same idea as the
                // cursor repulsion but centred on the mic icon and travelling
                // outward over time instead of following the pointer
                if (rings.length) {{
                  const ox = d.x - originX, oy = d.y - originY;
                  const odist = Math.hypot(ox, oy);
                  if (odist > 0.001) {{
                    for (const ring of rings) {{
                      const diff = Math.abs(odist - ring.r);
                      if (diff < RING_BAND) {{
                        const f = (1 - diff / RING_BAND) * ring.s;
                        px += (ox / odist) * f * 130;
                        py += (oy / odist) * f * 130;
                      }}
                    }}
                  }}
                }}

                // spring back toward home, with damping
                // Tuned by measurement: a stiffer spring here becomes underdamped
                // and oscillates, which reads as a *weaker* effect, not a stronger one.
                d.vx += ((tx + px) - d.x) * 0.045;
                d.vy += ((ty + py) - d.y) * 0.045;
                d.vx *= 0.82; d.vy *= 0.82;
                d.x += d.vx; d.y += d.vy;

                ctx.globalAlpha = d.a;
                ctx.fillStyle = d.c;
                ctx.beginPath();
                ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
                ctx.fill();
              }}
              ctx.globalAlpha = 1;
            }}

            cv.__dbg = {{mouse: mouse, dots: () => dots}};   // inspection hook
            build();
            frame();
            // A timer rather than requestAnimationFrame: this script runs inside a
            // zero-height iframe, where rAF is throttled or suspended entirely.
            // 30fps is ample for a slow drifting field.
            cv.__timer = setInterval(frame, 33);
          }} catch (e) {{ /* decoration only — never break the app */ }}
        }})();
        </script>
        """,
        height=0,
    )


def inject_micro_interactions():
    """Cursor-tracked spotlight + tilt on `.card` hover, and a click-position
    ripple on primary buttons.

    Event-driven only — a mousemove/click listener, no continuous animation
    loop — which is a lighter, simpler ask of the same same-origin-iframe
    mechanism inject_particles() already uses, just scoped to individual
    cards/buttons on hover/click rather than the whole page. Guarded so a
    Streamlit rerun re-executing this never binds a second copy of the
    listeners, and skips the tilt/spotlight entirely under reduced motion.
    """
    components.html(
        """
        <script>
        (function () {
          try {
            const doc = window.parent.document;
            if (doc.__vgMicroBound) return;
            doc.__vgMicroBound = true;

            const reduced = window.parent.matchMedia
              && window.parent.matchMedia('(prefers-reduced-motion: reduce)').matches;

            let lastCard = null;
            doc.addEventListener('mousemove', function (e) {
              const card = e.target.closest && e.target.closest('.card');
              if (card !== lastCard) {
                if (lastCard) {
                  lastCard.classList.remove('spotlit');
                  lastCard.style.transform = '';
                }
                lastCard = card;
                if (card) card.classList.add('spotlit');
              }
              if (card) {
                const r = card.getBoundingClientRect();
                const px = (e.clientX - r.left) / r.width;
                const py = (e.clientY - r.top) / r.height;
                card.style.setProperty('--mx', (px * 100).toFixed(1) + '%');
                card.style.setProperty('--my', (py * 100).toFixed(1) + '%');
                if (!reduced) {
                  const ry = (px - 0.5) * 6;
                  const rx = (0.5 - py) * 6;
                  card.style.transform =
                    'perspective(700px) rotateX(' + rx.toFixed(2) + 'deg) rotateY(' + ry.toFixed(2) + 'deg)';
                }
              }
            });

            doc.addEventListener('click', function (e) {
              const btn = e.target.closest
                && e.target.closest('div[data-testid="stButton"] button[kind="primary"]');
              if (!btn) return;
              const r = btn.getBoundingClientRect();
              const size = Math.max(r.width, r.height) * 1.6;
              const span = doc.createElement('span');
              span.className = 'vg-ripple';
              span.style.width = span.style.height = size + 'px';
              span.style.left = (e.clientX - r.left - size / 2) + 'px';
              span.style.top = (e.clientY - r.top - size / 2) + 'px';
              btn.appendChild(span);
              span.addEventListener('animationend', function () { span.remove(); });
            });
          } catch (e) { /* decoration only — never break the app */ }
        })();
        </script>
        """,
        height=0,
    )


def scroll_to_top():
    """Reset scroll position — called only on the rerun where the sidebar
    selection actually changed, and only after that page's own content has
    been added to the script (see the call site after the footer), never on
    every rerun, so moving a slider on the *same* page never yanks the view
    back up. Streamlit reuses the same scrollable container across reruns
    rather than giving each page its own, so without this a page that was
    scrolled down stays scrolled down when you switch to a different one.

    Keeps re-asserting scroll=0 for a short window afterward, not just once:
    even after this page's content is in the DOM, the document's height
    keeps changing slightly later, from things a MutationObserver can't see
    — an <img> finishing an async load doesn't mutate the DOM, only its
    rendered size — and the browser's own scroll anchoring quietly
    re-adjusts scroll position to compensate. Measured live, scroll actually
    drifted to over 1000px around 250ms after a page switch before settling
    back to 0 on its own. A polling loop catches that regardless of what
    caused it and stops after ~1.5s so it never fights a real scroll the
    user makes on the page afterward — setInterval, not
    requestAnimationFrame: this script runs inside a zero-height iframe,
    where rAF is throttled or suspended entirely (same reason
    inject_particles() below uses a timer instead of rAF).
    """
    components.html(
        """
        <script>
        (function () {
          try {
            const doc = window.parent.document;
            const main = doc.querySelector('section.stMain')
              || doc.querySelector('[data-testid="stAppViewContainer"]')
              || doc.querySelector('.main');

            function resetScroll() {
              try {
                if (main) main.scrollTo({ top: 0, left: 0, behavior: 'instant' });
                doc.documentElement.scrollTo(0, 0);
                doc.body.scrollTo(0, 0);
              } catch (e) { /* best-effort only — never break the app */ }
            }

            resetScroll();
            const start = Date.now();
            const timer = setInterval(() => {
              if (main && main.scrollTop !== 0) resetScroll();
              if (Date.now() - start > 1500) clearInterval(timer);
            }, 10);
          } catch (e) { /* best-effort only — never break the app */ }
        })();
        </script>
        """,
        height=0,
    )


# Staggered delays and durations so the equaliser bars never move in lockstep.
_EQ_BARS = "".join(
    f'<i style="animation-delay:{d}s;animation-duration:{s}s"></i>'
    for d, s in [(0.0, 1.10), (0.22, 0.86), (0.44, 1.28), (0.11, 0.98), (0.33, 1.16),
                 (0.55, 0.90), (0.17, 1.22), (0.39, 1.02), (0.06, 1.34), (0.28, 0.94),
                 (0.50, 1.18), (0.13, 1.06)]
)


def hero(pill, title, subtitle):
    st.markdown(
        f"""<div class="hero">
        <span class="pill">{pill}</span><span class="live-dot" title="You are here" aria-hidden="true"></span>
        <h1>{title}</h1><p>{subtitle}</p>
        <div class="eq">{_EQ_BARS}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def stat_row(items):
    cols = st.columns(len(items))
    for col, (value, key) in zip(cols, items):
        col.markdown(f"""<div class="stat"><div class="v">{value}</div><div class="k">{key}</div></div>""",
                     unsafe_allow_html=True)


def card_row(items):
    cols = st.columns(len(items))
    for col, (icon, head, body) in zip(cols, items):
        col.markdown(
            f"""<div class="card"><span class="ico">{icon}</span><h4>{head}</h4><p>{body}</p></div>""",
            unsafe_allow_html=True,
        )


def steps(items):
    for i, (head, body) in enumerate(items, start=1):
        st.markdown(
            f"""<div class="step"><div class="n">{i}</div>
            <div><h5>{head}</h5><p>{body}</p></div></div>""",
            unsafe_allow_html=True,
        )


def status_card_row(items, color):
    """Like card_row, but with a status-coloured left edge and tint —
    used on the Coverage page (confirmed-detects / confirmed-escapes / unknown)."""
    cols = st.columns(len(items))
    for col, (icon, head, body) in zip(cols, items):
        col.markdown(
            f"""<div class="card" style="border-left:3px solid {color};
                 background:linear-gradient(160deg,{color}14,{C['panel']});">
                <span class="ico">{icon}</span><h4>{head}</h4><p>{body}</p></div>""",
            unsafe_allow_html=True,
        )


def _flatten_markup(markup):
    """Collapse a pretty-printed multi-line SVG/HTML string onto one line.

    Streamlit's markdown renderer treats 4+ leading spaces as an indented
    code block rather than HTML passthrough — harmless when the string is
    the whole markdown call, but nesting one of these inside another
    f-string (a sparkline inside a stat tile, say) can reintroduce that much
    leading whitespace at the substitution point, and it silently renders as
    literal text instead of the SVG. Flattening removes the risk regardless
    of where the string ends up embedded.
    """
    return " ".join(s for line in markup.splitlines() if (s := line.strip()))


def gauge_svg(p_genuine, label, threshold=DEFAULT_THRESHOLD):
    """Semicircular gauge of the raw score, with the decision threshold marked.

    The colored arc draws itself in and the score counts up to its final
    value — both plain CSS keyframe animations (no JS), matching how the
    app's other motion (hero entrance, mic pulse) already works. Each gets
    a uniquely-numbered class/keyframe name (`uid`) so multiple gauges
    rendered in the same rerun — e.g. a multi-file upload — never fight
    over the same animation.
    """
    color = C["genuine"] if label == "Genuine" else C["synthetic"]
    r, cx, cy = 80, 100, 100

    def point(v, radius=r):
        rad = np.deg2rad(180 - 180 * v)
        return cx + radius * np.cos(rad), cy - radius * np.sin(rad)

    x, y = point(p_genuine)
    tx1, ty1 = point(threshold, r - 13)
    tx2, ty2 = point(threshold, r + 13)
    arc_len = np.pi * r * p_genuine
    score_int = round(p_genuine * 100)
    uid = next(_uid_counter)

    # A few stops on the way to the final score, so the number reads as
    # counting up rather than appearing instantly. Plain keyframes swapping
    # generated ::after content — no exotic CSS feature dependency, so it
    # holds up in any browser that runs CSS animations at all. The resting
    # (non-animated / reduced-motion) content is the real final score.
    stops = [round(score_int * f) for f in (0.0, 0.32, 0.55, 0.74, 0.9, 1.0)]
    count_frames = "\n".join(
        f'{pct}% {{ content: "{v}%"; }}'
        for pct, v in zip((0, 20, 40, 60, 80, 100), stops)
    )

    return _flatten_markup(f"""
    <svg viewBox="0 0 200 132" width="100%" style="max-width:270px">
      <path d="M 20 100 A {r} {r} 0 0 1 180 100" fill="none"
            stroke="{C['border']}" stroke-width="16" stroke-linecap="round"/>
      <path d="M 20 100 A {r} {r} 0 0 1 {x:.2f} {y:.2f}" fill="none"
            stroke="{color}" stroke-width="16" stroke-linecap="round"
            class="anim-reveal" style="stroke-dasharray:{arc_len:.2f};stroke-dashoffset:0;
            animation:gaugeArc{uid} .9s cubic-bezier(.16,.84,.44,1);"/>
      <line x1="{tx1:.2f}" y1="{ty1:.2f}" x2="{tx2:.2f}" y2="{ty2:.2f}"
            stroke="{C['text']}" stroke-width="2.5"/>
      <foreignObject x="30" y="56" width="140" height="36">
        <div xmlns="http://www.w3.org/1999/xhtml" class="gauge-score{uid} anim-reveal"
             style="display:flex;align-items:center;justify-content:center;height:100%;
             font-family:'JetBrains Mono',monospace;font-size:29px;font-weight:800;color:{color};"></div>
      </foreignObject>
      <text x="100" y="105" text-anchor="middle" fill="{C['muted']}"
            font-size="10" letter-spacing="1.1">SCORE — LIKELIHOOD GENUINE</text>
      <text x="100" y="126" text-anchor="middle" fill="{C['muted']}"
            font-size="10">threshold {threshold:.2f} (marked)</text>
      <style>
        .gauge-score{uid}::after {{
            content: "{score_int}%";
            animation: gaugeCount{uid} .9s ease-out;
        }}
        @keyframes gaugeCount{uid} {{ {count_frames} }}
        @keyframes gaugeArc{uid} {{ from {{ stroke-dashoffset:{arc_len:.2f}; }} to {{ stroke-dashoffset:0; }} }}
      </style>
    </svg>
    """)


def sparkline_svg(values, color):
    """Small inline trend line for a stat tile — real figures (e.g. a metric's
    value across the four training rounds), not decoration. Draws itself in
    the same self-drawing-arc style as the gauge, via stroke-dashoffset."""
    w, h, pad = 120, 30, 4
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    pts = []
    for i, v in enumerate(values):
        px = pad + i * (w - 2 * pad) / (len(values) - 1)
        py = h - pad - (v - lo) / span * (h - 2 * pad)
        pts.append((px, py))
    line = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    path_len = sum(
        np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    ) or 1
    lx, ly = pts[-1]
    uid = next(_uid_counter)
    return _flatten_markup(f"""
    <svg viewBox="0 0 {w} {h}" width="100%" height="{h}" style="display:block;margin-top:6px;">
      <path d="{line}" fill="none" stroke="{color}" stroke-width="2"
            stroke-linecap="round" stroke-linejoin="round" class="anim-reveal"
            style="stroke-dasharray:{path_len:.1f};stroke-dashoffset:0;
            animation:spark{uid} 1s cubic-bezier(.16,.84,.44,1);"/>
      <circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.6" fill="{color}" class="anim-reveal"
              style="opacity:1;animation:sparkdot{uid} .3s ease;animation-delay:.85s;
              animation-fill-mode:backwards;"/>
      <style>
        @keyframes spark{uid} {{ from {{ stroke-dashoffset:{path_len:.1f}; }} to {{ stroke-dashoffset:0; }} }}
        @keyframes sparkdot{uid} {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
      </style>
    </svg>
    """)


def upload_icon_svg():
    """Line-art upload glyph — replaces an emoji in the capture-prompt icon,
    same idea as the sparkline/gauge: a real icon rather than a placeholder
    character. A function (not a constant) so it re-reads the current
    theme's accent each time, not whichever theme was active at import."""
    return _flatten_markup(f"""
    <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="{C['accent']}"
         stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <path d="M12 3v10"/>
      <path d="M8 7l4-4 4 4"/>
      <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>
    </svg>
    """)


def mic_icon_svg():
    """Line-art microphone glyph — same rationale as upload_icon_svg()."""
    return _flatten_markup(f"""
    <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="{C['accent']}"
         stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <rect x="9" y="2" width="6" height="12" rx="3"/>
      <path d="M5 11a7 7 0 0 0 14 0"/>
      <path d="M12 18v3"/>
      <path d="M9 21h6"/>
    </svg>
    """)


def skeleton_card_html():
    """Shimmering placeholder shown for the moment a model is actually
    loading — a shape for that gap instead of blank space or the default
    spinner, reusing the app's own card look."""
    return _flatten_markup("""
    <div class="skeleton-card">
      <div class="skel-line" style="width:70%;"></div>
      <div class="skel-line" style="width:45%;"></div>
      <div class="skel-block"></div>
    </div>
    """)


def plot_analysis(y, sr, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), facecolor=C["panel"])
    for ax in (ax1, ax2):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["muted"], labelsize=8)
        for s in ax.spines.values():
            s.set_color(C["border"])

    librosa.display.waveshow(y, sr=sr, ax=ax1, color=C["accent"])
    ax1.set_title(f"Waveform — {title}", color=C["text"], fontsize=10)
    ax1.set_xlabel("")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax2, cmap="magma")
    ax2.set_title("Mel Spectrogram — frequency energy over time", color=C["text"], fontsize=10)
    cb = fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    cb.ax.yaxis.set_tick_params(color=C["muted"], labelsize=7)
    plt.setp(cb.ax.get_yticklabels(), color=C["muted"])

    plt.tight_layout()
    return fig


def plot_mfcc(mfcc):
    fig, ax = plt.subplots(figsize=(9, 2.6), facecolor=C["panel"])
    ax.set_facecolor(C["panel"])
    img = librosa.display.specshow(mfcc, x_axis="time", sr=SAMPLE_RATE, ax=ax, cmap="viridis")
    ax.set_title("MFCC — the 40-band 'voice fingerprint' fed to the network",
                 color=C["text"], fontsize=10)
    ax.tick_params(colors=C["muted"], labelsize=8)
    for s in ax.spines.values():
        s.set_color(C["border"])
    cb = fig.colorbar(img, ax=ax)
    cb.ax.yaxis.set_tick_params(color=C["muted"], labelsize=7)
    plt.setp(cb.ax.get_yticklabels(), color=C["muted"])
    plt.tight_layout()
    return fig


def result_card(filename, label, margin, duration, sr, model_name, threshold, p_genuine):
    filename = html.escape(filename)  # user-supplied, rendered as raw HTML below
    color = C["genuine"] if label == "Genuine" else C["synthetic"]
    icon = "✅" if label == "Genuine" else "⚠️"
    verdict = "Genuine human voice" if label == "Genuine" else "AI-generated / cloned voice"
    rel = "above" if label == "Genuine" else "at or below"
    thr_note = (f"Score {p_genuine:.3f} is {rel} the {threshold:.2f} threshold."
                + ("" if abs(threshold - DEFAULT_THRESHOLD) < 1e-9
                   else f" (default is {DEFAULT_THRESHOLD:.2f})"))
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{color}1f,{C['panel']});
                    border:1px solid {color}55;border-radius:16px;padding:20px 24px;margin:6px 0 14px 0;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">
            <div>
              <div style="color:{C['muted']};font-size:11px;letter-spacing:.9px;text-transform:uppercase;">{filename}</div>
              <div style="color:{color};font-size:24px;font-weight:800;margin-top:4px;">{icon} {label}</div>
              <div style="color:{C['muted']};font-size:13px;margin-top:2px;">{verdict}</div>
            </div>
            <div style="text-align:right;">
              <div style="color:{C['text']};font-size:26px;font-weight:800;">{margin*100:.1f}%</div>
              <div style="color:{C['muted']};font-size:11px;">margin from threshold</div>
            </div>
          </div>
          <div style="background:{C['border']};border-radius:6px;height:8px;margin-top:16px;overflow:hidden;">
            <div style="background:{color};width:{margin*100:.1f}%;height:100%;"></div>
          </div>
          <div style="color:{C['muted']};font-size:12px;margin-top:12px;">{thr_note}</div>
          <div style="color:{C['muted']};font-size:12px;margin-top:4px;">
            Duration {duration:.2f}s &nbsp;•&nbsp; {sr} Hz &nbsp;•&nbsp; Model: {model_name}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


FIGURES = [
    ("roc_curves.png", "ROC curves",
     "Each curve traces the trade-off between catching fakes and wrongly flagging real voices, "
     "across every possible decision threshold. A curve hugging the top-left is strong; the diagonal "
     "is random guessing. The baseline's real-world curve (AUC 0.625) sits close to that diagonal — "
     "visual proof it could barely discriminate on everyday recordings."),
    ("det_curves.png", "DET curves",
     "The format anti-spoofing research conventionally uses: the two error rates plotted against each "
     "other, with the dotted diagonal marking equal error. Where a curve crosses it is that model's EER. "
     "Note the crossover on ASVspoof — at low false-acceptance rates, the security-conscious operating "
     "region, the combined model outperforms the baseline."),
    ("confusion_matrices.png", "Confusion matrices",
     "Raw outcome counts at the default 0.5 threshold. The bottom-left cell of each is the dangerous "
     "one: forgeries accepted as genuine. On ASVspoof the combined model admits just 76, against the "
     "baseline's 205 and the fine-tuned model's 5,850."),
]


def threshold_stance(thr):
    if thr > 0.75:
        return ("Strict — prioritises catching forgeries, at the cost of more genuine "
                "voices being flagged for review.")
    if thr < 0.25:
        return ("Lenient — prioritises not accusing real people, at the cost of more "
                "forgeries slipping through.")
    return "Balanced — roughly even weighting of the two error types."


def render_figures():
    fig_dir = "figures"
    present = [(f, t, d) for f, t, d in FIGURES if os.path.exists(os.path.join(fig_dir, f))]
    if not present:
        return
    st.markdown("### Curves and matrices")
    st.caption("Generated on the same held-out evaluation sets used for the tables above.")
    tabs = st.tabs([t for _, t, _ in present])
    for tab, (fname, title, desc) in zip(tabs, present):
        with tab:
            st.image(os.path.join(fig_dir, fname), use_container_width=True)
            st.caption(desc)


# ------------------------------------------------------------ history storage
# Signed-in users' history lives in users.db (auth.py) and survives logout.
# Guests get the same Session Report experience, but only for this browser
# session — held in memory, never written to disk, so a public deployment
# doesn't keep strangers' recordings.
GUEST_HISTORY_LIMIT = 20  # caps per-guest memory on a shared free-tier server


def is_guest():
    return bool(st.session_state.user and st.session_state.user.get("is_guest"))


def history_add(filename, prediction, score, threshold, model, margin, analysed_at, audio_bytes):
    if is_guest():
        st.session_state.guest_records.append({
            "id": f"g{uuid.uuid4().hex[:10]}",
            "filename": filename, "prediction": prediction, "score": score,
            "threshold": threshold, "model": model, "margin": margin,
            "analysed_at": analysed_at, "audio_bytes": audio_bytes,
        })
        del st.session_state.guest_records[:-GUEST_HISTORY_LIMIT]
    else:
        auth.add_analysis_record(st.session_state.user["id"], filename, prediction, score,
                                 threshold, model, margin, analysed_at, audio_bytes=audio_bytes)


def history_records():
    if is_guest():
        return list(st.session_state.guest_records)
    return auth.get_analysis_records(st.session_state.user["id"])


def history_count():
    if is_guest():
        return len(st.session_state.guest_records)
    return auth.count_analysis_records(st.session_state.user["id"])


def history_clear():
    if is_guest():
        st.session_state.guest_records = []
    else:
        auth.clear_analysis_records(st.session_state.user["id"])


def history_delete(record_id):
    if is_guest():
        st.session_state.guest_records = [
            r for r in st.session_state.guest_records if r["id"] != record_id
        ]
    else:
        auth.delete_analysis_record(st.session_state.user["id"], record_id)


def record_audio(r):
    if "audio_bytes" in r:
        return r["audio_bytes"]
    return auth.get_audio_bytes(r.get("audio_path"))


def analyse(model, model_name, filename, raw_bytes, show_details=True,
            threshold=DEFAULT_THRESHOLD, show_playback=True):
    if show_playback:
        st.audio(raw_bytes)

    with st.status(f"Analysing {filename}…", expanded=True) as status:
        status.write("🎧 Loading and preparing audio…")
        y, sr, err = load_audio_safely(raw_bytes)
        if err:
            status.update(label="Could not analyse this clip", state="error")
            st.error(f"**{filename}** — {err}")
            return

        status.write("📊 Extracting MFCC features and running the CNN + BiLSTM model…")
        label, margin, p_genuine, mfcc = predict(model, y, sr, threshold)

        status.write("✅ Done")
        status.update(label=f"Analysis complete — {filename}", state="complete")

    # Saved before the result card and plots render, so leaving the page while
    # they're still drawing can't drop the record.
    history_add(
        filename, label, f"{p_genuine:.3f}", f"{threshold:.2f}", model_name, f"{margin:.3f}",
        now_local().strftime("%Y-%m-%d %H:%M:%S"), audio_bytes=raw_bytes,
    )

    left, right = st.columns([1, 1.6])
    with left:
        st.markdown(gauge_svg(p_genuine, label, threshold), unsafe_allow_html=True)
    with right:
        result_card(filename, label, margin, len(y) / sr, sr, model_name,
                    threshold, p_genuine)

    if show_details:
        with st.expander("🔬 Show the signal analysis behind this result"):
            fig = plot_analysis(y, sr, filename)
            st.pyplot(fig)
            plt.close(fig)
            fig2 = plot_mfcc(mfcc)
            st.pyplot(fig2)
            plt.close(fig2)



# ============================================================
# PAGES
# ============================================================
def page_overview(models):
    hero("Fraud prevention • Audio forensics",
         "Synthetic Voice Detection",
         "An end-to-end system that listens to a voice clip and determines whether it came from a "
         "real human or an AI voice-cloning engine — built to counter voice-based scams and support "
         "forensic analysis.")

    st.markdown("### Why this matters")
    card_row([
        ("🎭", "Cloning is trivial now",
         "Modern tools can clone a convincing voice from under a minute of reference audio, freely available online."),
        ("💸", "Real financial harm",
         "Attackers impersonate family members and company executives over calls to authorise fraudulent transfers."),
        ("👂", "Humans can't reliably tell",
         "High-quality clones routinely fool human listeners, which is exactly why an automated detector is needed."),
    ])

    st.markdown("### System at a glance")
    stat_row([
        ("CNN + BiLSTM", "Architecture"),
        ("40 × 400", "MFCC input"),
        ("ASVspoof 2019", "Base dataset"),
        (f"{len(models)}", "Models loaded"),
    ])

    st.markdown("### Detection pipeline")
    st.markdown(
        f"""
        <svg viewBox="0 0 900 120" width="100%" style="margin:6px 0 10px 0;">
          <defs>
            <linearGradient id="g1" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="{C['accent']}"/><stop offset="100%" stop-color="{C['accent2']}"/>
            </linearGradient>
            <marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3"
                    orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L7,3 z" fill="{C['muted']}"/>
            </marker>
          </defs>
          {''.join(
            f'''<g>
              <rect x="{18 + i*178}" y="30" width="150" height="58" rx="12"
                    fill="{C['panel']}" stroke="url(#g1)" stroke-width="1.4"/>
              <text x="{93 + i*178}" y="55" text-anchor="middle" fill="{C['text']}"
                    font-size="13" font-weight="700">{t}</text>
              <text x="{93 + i*178}" y="72" text-anchor="middle" fill="{C['muted']}"
                    font-size="10.5">{s}</text>
            </g>
            <line x1="{170 + i*178}" y1="59" x2="{194 + i*178}" y2="59"
                  stroke="{C['muted']}" stroke-width="1.6" marker-end="url(#ar)"/>'''
            for i, (t, s) in enumerate([
                ("Capture", "upload or record"),
                ("Pre-process", "16 kHz mono"),
                ("MFCC", "40 × 400 features"),
                ("CNN + BiLSTM", "neural network"),
            ])
          )}
          <g>
            <rect x="730" y="30" width="150" height="58" rx="12"
                  fill="{C['genuine']}18" stroke="{C['genuine']}" stroke-width="1.4"/>
            <text x="805" y="55" text-anchor="middle" fill="{C['text']}"
                  font-size="13" font-weight="700">Verdict</text>
            <text x="805" y="72" text-anchor="middle" fill="{C['muted']}"
                  font-size="10.5">genuine / synthetic</text>
          </g>
        </svg>
        """,
        unsafe_allow_html=True,
    )

    st.info("Open **Detect Voice** in the sidebar to run a live analysis.")


def page_detect(models):
    hero("Live analysis", "Detect Voice",
         "Upload existing audio files or record directly from your microphone. Every clip stays playable "
         "next to its verdict, so results can be replayed and verified on the spot.")

    if not models:
        st.error(f"No model file found. Place `{BASELINE_MODEL}` in the app folder and reload.")
        return

    names = list(models.keys())
    default_idx = len(names) - 1
    col1, col2 = st.columns([2, 1])
    with col1:
        chosen = st.selectbox("Detection model", names, index=default_idx,
                              help="Compare the original and fine-tuned models on the same clip.")
    with col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        details = st.checkbox("Show signal analysis", value=True)

    # load_model() is @st.cache_resource — instant on every switch back to an
    # already-loaded model, and genuinely brief even on a cold load. The
    # skeleton still gives that first moment a shape instead of a blank gap.
    model_slot = st.empty()
    model_slot.markdown(skeleton_card_html(), unsafe_allow_html=True)
    model = load_model(models[chosen])
    model_slot.empty()
    chosen_path = models[chosen]

    note = MODEL_NOTES.get(chosen_path)
    scores = MODEL_SCORES.get(chosen_path)
    if note and scores:
        asv, itw = scores["asv"], scores["itw"]
        st.markdown(
            f"""<div style="background:{C['panel']};border:1px solid {C['border']};
                 border-radius:14px;padding:14px 18px;margin:2px 0 12px 0;">
              <div style="color:{C['muted']};font-size:13px;line-height:1.55;">{note}</div>
              <div style="display:flex;gap:26px;margin-top:10px;flex-wrap:wrap;">
                <div><span style="color:{C['muted']};font-size:11px;">STUDIO AUDIO (ASVspoof)</span><br/>
                  <b style="color:{C['text']};font-size:15px;">{asv[0]*100:.1f}%</b>
                  <span style="color:{C['muted']};font-size:11.5px;"> acc &nbsp;·&nbsp;
                  {asv[2]*100:.1f}% prec</span></div>
                <div><span style="color:{C['muted']};font-size:11px;">REAL-WORLD (In-the-Wild)</span><br/>
                  <b style="color:{C['text']};font-size:15px;">{itw[0]*100:.1f}%</b>
                  <span style="color:{C['muted']};font-size:11.5px;"> acc &nbsp;·&nbsp;
                  {itw[2]*100:.1f}% prec</span></div>
                <div><span style="color:{C['muted']};font-size:11px;">EER (studio / real-world)</span><br/>
                  {f'<b style="color:{C["text"]};font-size:15px;">{scores["eer"][0]*100:.2f}%</b>'
                   f'<span style="color:{C["muted"]};font-size:11.5px;"> / </span>'
                   f'<b style="color:{C["text"]};font-size:15px;">{scores["eer"][1]*100:.2f}%</b>'
                   if scores.get("eer") else
                   f'<span style="color:{C["muted"]};font-size:13px;">not yet measured</span>'}</div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("⚖️  Decision threshold"):
        st.caption(
            "The model outputs a score from 0 to 1 for how likely a voice is genuine. The threshold "
            "is the cut-off above which it is declared Genuine. Raising it makes the system stricter — "
            "it catches more forgeries but flags more real people; lowering it does the reverse. "
            "There is no universally correct value: it depends on whether a missed forgery or a false "
            "accusation is the costlier mistake."
        )
        sc1, sc2 = st.columns([3, 1])
        # Reset works by bumping a nonce so the slider gets a fresh key and is
        # rebuilt at its default. Streamlit otherwise retains widget state across
        # reruns, which would silently ignore the reset.
        with sc2:
            if st.button("Reset to 0.50", use_container_width=True):
                st.session_state.thr_nonce += 1
                st.rerun()
        with sc1:
            thr = st.slider("Threshold", 0.01, 0.99, DEFAULT_THRESHOLD, 0.01,
                            key=f"thr_{st.session_state.thr_nonce}",
                            label_visibility="collapsed")
        st.session_state.threshold = thr
        st.markdown(f"""<div class="banner"><b>{thr:.2f} · {threshold_stance(thr)}</b></div>""",
                    unsafe_allow_html=True)

        if chosen_path in EER_THRESHOLDS:
            st.caption(
                f"For reference, this model's equal-error point on ASVspoof falls at a threshold of "
                f"{EER_THRESHOLDS[chosen_path]:.4f}. The four models are calibrated "
                "very differently, which is why a single fixed cut-off suits none of them perfectly."
            )

    threshold = thr
    if abs(threshold - DEFAULT_THRESHOLD) > 1e-9:
        st.caption(f"⚖️ Active decision threshold: **{threshold:.2f}** — {threshold_stance(threshold)} "
                   f"(default {DEFAULT_THRESHOLD:.2f})")

    if len(names) > 1:
        st.markdown(
            f"""<div class="banner"><b>Comparison tip.</b> Run the same clip through each model in
            turn — "{names[0]}" then "{names[-1]}" — to demonstrate the improvement live. Every run is
            logged in the Session Report.</div>""",
            unsafe_allow_html=True,
        )

    mode = option_menu(
        menu_title=None,
        options=["Upload Audio", "Record Live"],
        icons=["cloud-arrow-up-fill", "mic-fill"],
        orientation="horizontal",
        default_index=0,
        key="input_mode",
        styles={
            "container": {
                "padding": "5px", "background-color": C["panel"],
                "border": f"1px solid {C['border']}", "border-radius": "14px",
            },
            "icon": {"color": C["accent"], "font-size": "16px"},
            "nav-link": {
                "font-size": "13.5px", "text-align": "center", "padding": "12px 10px",
                "border-radius": "10px", "color": C["muted"], "margin": "0 3px",
                "font-weight": "600",
            },
            "nav-link-selected": {
                "background": f"linear-gradient(135deg,{C['accent']},{C['accent2']})",
                "color": C["bg"], "font-weight": "700",
            },
        },
    )

    with st.container(border=True):
        if mode == "Upload Audio":
            st.markdown(
                f"""<div class="io-caption">
                  <div class="upload-glyph">{upload_icon_svg()}</div>
                  <div class="t">Drop a clip to analyse</div>
                  <div class="s">WAV · MP3 · FLAC — multiple files supported</div>
                </div>""",
                unsafe_allow_html=True,
            )
            files = st.file_uploader("Audio files", type=["wav", "mp3", "flac"],
                                     accept_multiple_files=True, label_visibility="collapsed")
            if files:
                # A fresh file (or set of files) always needs a fresh explicit
                # analyse click — same gate the recording flow already has.
                # Re-running analyse() on every later rerun (e.g. moving the
                # threshold slider) is still fine and intended, since that's
                # not a new upload.
                file_sig = tuple((f.name, f.size) for f in files)
                if st.session_state.upload_sig != file_sig:
                    st.session_state.upload_sig = file_sig
                    st.session_state.upload_analysed = False

                for f in files:
                    st.divider()
                    st.markdown(f"#### 📄 {f.name}")
                    st.audio(f.getvalue())

                st.divider()
                if st.button("🔍  Analyze Voice", use_container_width=True, type="primary"):
                    st.session_state.upload_analysed = True

                if st.session_state.upload_analysed:
                    for f in files:
                        st.divider()
                        st.markdown(f"#### Result — {f.name}")
                        analyse(model, chosen, f.name, f.getvalue(), details, threshold,
                                show_playback=False)
            else:
                st.session_state.upload_analysed = False
                st.info("Upload one or more clips to analyse them.")
        else:
            pending = st.session_state.pending_recording
            st.markdown(
                f"""<div class="io-caption">
                  <div class="mic-orb">{mic_icon_svg()}</div>
                  <div class="t">{"Speak, then stop — you'll get to hear it back first" if not pending else "Happy with the take?"}</div>
                  <div class="s">5-10 seconds works best</div>
                </div>""",
                unsafe_allow_html=True,
            )
            audio = mic_recorder(start_prompt="⏺️  Start recording", stop_prompt="⏹️  Stop recording",
                                 just_once=False, use_container_width=True,
                                 key=f"rec_{st.session_state.rec_nonce}")
            # A fresh recording (bytes differ from whatever is already pending) enters
            # review — it is NOT analysed yet. mic_recorder keeps returning its last
            # result on every rerun, so bytes must be compared against the raw
            # capture, not just truthiness.
            if audio and (pending is None or audio["bytes"] != pending.get("raw_bytes")):
                # Decode + peak-normalize immediately, so the review playback is
                # already at an audible, model-consistent level — not the raw,
                # often very quiet, browser-microphone capture (see
                # load_audio_safely for why quiet audio also skews the score).
                y, sr, err = load_audio_safely(audio["bytes"])
                if err:
                    st.error(f"Could not process that recording — {err}")
                else:
                    buf = io.BytesIO()
                    sf.write(buf, y, sr, format="WAV")
                    st.session_state.pending_recording = {
                        "raw_bytes": audio["bytes"],
                        "bytes": buf.getvalue(),
                        "name": f"live_recording_{now_local():%H%M%S}.wav",
                    }
                    st.session_state.recording_analysed = False
                    st.rerun()

            pending = st.session_state.pending_recording
            if pending:
                st.audio(pending["bytes"])
                c1, c2 = st.columns(2)
                with c1:
                    discard = st.button("✖️  Discard & re-record", use_container_width=True)
                with c2:
                    confirm = st.button("🔍  Detect Voice", use_container_width=True, type="primary")
                if discard:
                    st.session_state.pending_recording = None
                    st.session_state.recording_analysed = False
                    st.session_state.rec_nonce += 1  # forces mic_recorder to reset its own state
                    st.rerun()
                if confirm:
                    st.session_state.recording_analysed = True
                if st.session_state.recording_analysed:
                    st.divider()
                    st.markdown(f"#### 🎤 {pending['name']}")
                    analyse(model, chosen, pending["name"], pending["bytes"], details, threshold,
                            show_playback=False)


def page_how():
    hero("Technical walkthrough", "How It Works",
         "From raw sound to a verdict in four stages — each one is inspectable, which makes the "
         "system explainable rather than a black box.")

    steps([
        ("Capture and normalise",
         "Audio arrives as an upload or a live microphone recording. It is converted to mono and resampled "
         "to 16 kHz so that every clip reaches the model in an identical format, regardless of the device "
         "it was recorded on."),
        ("Extract MFCC features",
         "The waveform is converted into Mel-Frequency Cepstral Coefficients — 40 bands over 400 time frames. "
         "MFCCs approximate how human hearing perceives sound, and they expose the subtle spectral artefacts "
         "that synthesis engines leave behind but human ears tend to miss."),
        ("Convolutional feature learning",
         "Two convolutional layers scan the MFCC map for local patterns — unnatural harmonic spacing, "
         "over-smoothed transitions, missing micro-variations — the fingerprints of machine-generated speech."),
        ("Temporal modelling and decision",
         "A bidirectional LSTM reads the sequence forwards and backwards, because authenticity cues appear in "
         "how a voice evolves over time. Averaging across all timesteps yields one probability, which becomes "
         "the final Genuine / Synthetic verdict."),
    ])

    st.markdown("### Network architecture")
    _arch_items = [
        ("Input", "1×40×400", 86, C['accent']),
        ("Conv2D", "16 filters", 74, C['accent']),
        ("MaxPool", "÷2", 62, C['accent']),
        ("Conv2D", "32 filters", 62, C['accent2']),
        ("MaxPool", "÷2", 50, C['accent2']),
        ("BiLSTM", "64×2 hidden", 74, C['accent2']),
        ("Dense", "32 units", 44, C['genuine']),
        ("Output", "genuine / synthetic", 34, C['genuine']),
    ]
    # viewBox width is derived from the item count, not hardcoded — a fixed
    # width previously clipped the last box once an item was added and the
    # layout outgrew it silently (only visible once there were enough boxes).
    _arch_spacing, _arch_box_w, _arch_left_pad, _arch_right_pad = 126, 104, 14, 14
    _arch_width = _arch_left_pad + (len(_arch_items) - 1) * _arch_spacing + _arch_box_w + _arch_right_pad
    st.markdown(
        f"""
        <svg viewBox="0 0 {_arch_width} 150" width="100%">
          {''.join(
            f'''<g>
              <rect x="{_arch_left_pad + i*_arch_spacing}" y="{56 - h/2}" width="{_arch_box_w}" height="{h}" rx="10"
                    fill="{col}22" stroke="{col}" stroke-width="1.3"/>
              <text x="{_arch_left_pad + _arch_box_w/2 + i*_arch_spacing}" y="60" text-anchor="middle" fill="{C['text']}"
                    font-size="11.5" font-weight="700">{t}</text>
              <text x="{_arch_left_pad + _arch_box_w/2 + i*_arch_spacing}" y="128" text-anchor="middle" fill="{C['muted']}" font-size="10">{s}</text>
            </g>'''
            for i, (t, s, h, col) in enumerate(_arch_items)
          )}
        </svg>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Reading the output")
    card_row([
        ("✅", "Genuine",
         "The model judges the recording more consistent with a real human vocal tract than with synthesis."),
        ("⚠️", "Synthetic",
         "Spectral or temporal artefacts characteristic of text-to-speech or voice-cloning engines were detected."),
        ("📊", "Confidence",
         "How strongly the model leans toward its verdict. Lower values mean a borderline case worth a second look."),
    ])

    st.markdown(
        f"""<div class="banner"><b>Honest limitation.</b> No detector is perfect. Very short clips,
        heavy background noise, or synthesis methods unlike anything in the training data can all cause
        mistakes. Confidence is a decision aid, not proof.</div>""",
        unsafe_allow_html=True,
    )


def page_model(models):
    hero("Training and evaluation", "Model & Results",
         "How the detector was trained, what it scores, and the real-world generalisation problem "
         "that shaped the current version.")

    # Worst→best order the four training rounds actually happened in — reused
    # below for the comparison table too, so both sections read the same story.
    order = [BASELINE_MODEL, FINETUNED_MODEL, COMBINED_MODEL, COMBINED_V2_MODEL]
    stage_labels = {
        BASELINE_MODEL: "1 · Baseline (ASVspoof only)",
        FINETUNED_MODEL: "2 · Fine-tuned (In-the-Wild only)",
        COMBINED_MODEL: "3 · Combined (both datasets)",
        COMBINED_V2_MODEL: "4 · Combined + classical TTS",
    }

    st.markdown("### Final model — measured on both domains")
    fin = MODEL_SCORES[COMBINED_V2_MODEL]
    tiles = [
        ("Studio accuracy", f"{fin['asv'][0]*100:.1f}%",
         [MODEL_SCORES[m]["asv"][0] * 100 for m in order]),
        ("Real-world accuracy", f"{fin['itw'][0]*100:.1f}%",
         [MODEL_SCORES[m]["itw"][0] * 100 for m in order]),
        ("Studio precision", f"{fin['asv'][2]*100:.1f}%",
         [MODEL_SCORES[m]["asv"][2] * 100 for m in order]),
        ("Real-world precision", f"{fin['itw'][2]*100:.1f}%",
         [MODEL_SCORES[m]["itw"][2] * 100 for m in order]),
    ]
    cols = st.columns(len(tiles))
    for col, (label, value, series) in zip(cols, tiles):
        col.markdown(
            f"""<div class="stat">
                  <div class="v">{value}</div>
                  <div class="k">{label}</div>
                  {sparkline_svg(series, C['accent'])}
                </div>""",
            unsafe_allow_html=True,
        )
    st.caption("Combined + classical-TTS model, evaluated on ASVspoof 2019 LA dev (24,844 studio "
               "clips) and In-the-Wild validation (6,355 real-world clips). Each sparkline traces "
               "that metric across all four training rounds (baseline → fine-tuned → combined → "
               "combined+TTS).")

    st.markdown("### Four-stage comparison")
    st.caption("Every model evaluated on both held-out test sets. Bona-fide = genuine human voice.")

    def _fmt(triple):
        return f"{triple[0]*100:.1f}% / {triple[1]*100:.1f}% / {triple[2]*100:.1f}%"

    st.table({
        "Model": [stage_labels[m] for m in order],
        "Studio — acc / recall / precision": [_fmt(MODEL_SCORES[m]["asv"]) for m in order],
        "Real-world — acc / recall / precision": [_fmt(MODEL_SCORES[m]["itw"]) for m in order],
    })

    st.markdown("### Equal Error Rate")
    st.caption("EER is threshold-independent: the rate at which false acceptances and false "
               "rejections are equal. Lower is better; 50% would be random guessing. Measured "
               "for all four models.")
    eer_order = [m for m in order if MODEL_SCORES[m].get("eer")]
    st.table({
        "Model": [stage_labels[m] for m in eer_order],
        "ASVspoof EER": [f"{MODEL_SCORES[m]['eer'][0]*100:.2f}%" for m in eer_order],
        "In-the-Wild EER": [f"{MODEL_SCORES[m]['eer'][1]*100:.2f}%" for m in eer_order],
    })
    st.markdown(
        f"""<div class="banner"><b>Reading the EER table.</b> The baseline's real-world EER of 41.52%
        is close to the 50% of random guessing — on everyday recordings it could not separate genuine
        from cloned speech at <i>any</i> threshold, so retraining was necessary rather than optional.
        Joint training cost 2.64 points of ASVspoof EER (5.83% → 8.47%) and returned 37.96 points of
        real-world EER (41.52% → 3.56%), roughly a twelvefold improvement in deployment conditions.
        Model 4 lands within half a point of model 3 on both domains (8.33% / 3.60%) — the
        classical-TTS fix didn't come at the cost of this equal-error balance.</div>""",
        unsafe_allow_html=True,
    )

    card_row([
        ("1️⃣", "Baseline fails in the real world",
         "97.6% on studio audio, but only 23.0% recall on real-world genuine voices — it labelled "
         "roughly three of every four real people as synthetic."),
        ("2️⃣", "Naive fine-tuning forgets",
         "Real-world accuracy jumped to 95.6%, but studio precision collapsed from 91.3% to 29.5% — "
         "the model began passing spoofed clips through as genuine."),
        ("3️⃣", "Joint training resolves both",
         "Training on both datasets together holds studio accuracy at 97.4% while reaching 96.5% "
         "on real-world audio. No forgetting."),
        ("4️⃣", "Classical TTS gap found, then closed",
         "Testing against Windows SAPI/espeak-ng — a synthesis method in neither dataset — found "
         "model 3 missed 2 of 3 samples. Adding 120 such clips to training fixed it (3 of 3), with "
         "accuracy/precision on the other two domains essentially unchanged."),
    ])

    st.markdown(
        f"""<div class="banner"><b>Reported trade-off (model 3 vs. baseline).</b> The combined model's
        studio bona-fide recall is 77.9%, below the baseline's 84.7%, so slightly more genuine studio
        clips are flagged for review. In exchange, precision rose from 91.3% to 96.3% — markedly fewer
        spoofs slip through. For a fraud detection tool this is the safer direction: a false alarm costs
        a second listen, a missed forgery costs money. Model 4 sits within a couple of points of model 3
        on both domains — recall 81.5% studio / 95.7% real-world, precision 94.2% studio / 98.1%
        real-world — so this trade did not meaningfully worsen when the classical-TTS fix was added.</div>""",
        unsafe_allow_html=True,
    )

    render_figures()

    st.markdown("### Training configuration")
    st.table({
        "Setting": ["Architecture", "Input features", "Optimiser", "Loss function",
                    "Class imbalance", "Training data", "Learning rate", "Epochs",
                    "Training hardware"],
        "Value": ["CNN (2 conv blocks) + bidirectional LSTM",
                  "40 MFCC × 400 frames, normalised (mean −3.155, std 29.654)",
                  "Adam", "Binary cross-entropy with logits",
                  "WeightedRandomSampler over the merged set (31,071 spoof vs 16,554 bona-fide)",
                  "ASVspoof 2019 LA (25,380) + In-the-Wild (22,245) = 47,625 clips",
                  "2e-5, initialised from the ASVspoof baseline checkpoint",
                  "6", "NVIDIA T4 GPU (Google Colab)"],
    })

    st.markdown("### Case study — the domain-mismatch problem")
    st.markdown(
        f"""<div class="banner"><b>Observed failure.</b> A genuine voice recorded casually on a laptop
        microphone was confidently misclassified as <i>Synthetic</i>. The model was not broken — it had
        simply never seen anything but clean studio audio, so ordinary microphone noise and compression
        looked abnormal to it.</div>""",
        unsafe_allow_html=True,
    )
    steps([
        ("Diagnosis",
         "Classic domain mismatch: training data (ASVspoof studio recordings) and deployment data "
         "(laptop microphone, MP3 compression, room noise) come from different distributions."),
        ("First remedy — and its hidden cost",
         "Continued training on the public In-the-Wild deepfake dataset lifted real-world accuracy from "
         "46.8% to 95.6%. Re-testing on the original ASVspoof data then revealed catastrophic forgetting: "
         "studio precision had collapsed from 91.3% to 29.5%, meaning the adapted model was passing "
         "spoofed clips through as genuine."),
        ("Second remedy — joint training",
         "Rather than fine-tuning on the new data alone, the model was retrained from the original "
         "checkpoint on both datasets simultaneously, so neither distribution could be forgotten. A low "
         "learning rate (2e-5) and a WeightedRandomSampler across the merged 47,625-clip set handled "
         "adaptation and class imbalance together."),
        ("Verification on both domains",
         "Every model was evaluated on both held-out test sets, not just the one it was trained for. "
         "This is what exposed the forgetting in the first place, and what confirms the combined model "
         "genuinely resolves it."),
    ])

    st.markdown("### Models currently available")
    if models:
        for name, path in models.items():
            size_kb = os.path.getsize(path) / 1024
            st.markdown(
                f"""<div class="step"><div class="n">✓</div><div>
                <h5>{name}</h5><p><code>{path}</code> — {size_kb:.0f} KB</p></div></div>""",
                unsafe_allow_html=True,
            )
    missing = [m for m in (BASELINE_MODEL, FINETUNED_MODEL, COMBINED_MODEL, COMBINED_V2_MODEL)
               if m not in models.values()]
    if missing:
        st.warning("Not yet in the app folder: " + ", ".join(f"`{m}`" for m in missing) +
                   " — add them to enable the full four-stage comparison.")


def _record_margin(r):
    """Older rows saved before the margin column existed fall back to
    recomputing it from the stored score/threshold, via the same formula
    predict() uses — new rows already have it stored directly."""
    if r.get("margin") not in (None, ""):
        return float(r["margin"])
    p_genuine, threshold = float(r["score"]), float(r["threshold"])
    if p_genuine > threshold:
        m = (p_genuine - threshold) / max(1.0 - threshold, 1e-9)
    else:
        m = (threshold - p_genuine) / max(threshold, 1e-9)
    return min(max(m, 0.0), 1.0)


REPORT_PAGE_SIZE = 9


@st.cache_data(show_spinner=False, max_entries=256)
def clip_profile(audio_bytes, n_bars):
    """Loudness envelope (n_bars peaks scaled 0-1) and duration of a stored
    clip. Cached on the audio bytes, so the recordings library decodes each
    clip once rather than on every rerun."""
    y, sr, err = load_audio_safely(audio_bytes)
    if err:
        return None
    env = np.array([c.max() if len(c) else 0.0 for c in np.array_split(np.abs(y), n_bars)])
    env = env / (env.max() or 1.0)
    return {"bars": [float(v) for v in env], "duration": len(y) / sr}


def waveform_svg(bars, color, height=44):
    """Mirrored bar waveform drawn from the clip's real loudness envelope —
    each bar grows in with a slight stagger, same CSS-only motion approach
    (and per-render uid) as the gauge and sparklines."""
    uid = next(_uid_counter)
    mid = height / 2
    rects = "".join(
        f'<rect x="{i * 4 + 0.7:.1f}" y="{mid - h / 2:.1f}" width="2.6" height="{h:.1f}" rx="1.3" '
        f'class="wf{uid} anim-reveal" style="animation-delay:{i * 0.01:.2f}s"/>'
        for i, h in ((i, max(2.0, v * (height - 4))) for i, v in enumerate(bars))
    )
    return _flatten_markup(f"""
    <svg viewBox="0 0 {len(bars) * 4} {height}" width="100%" height="{height}"
         preserveAspectRatio="none" style="display:block;">
      <defs><linearGradient id="wfg{uid}" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0" stop-color="{color}" stop-opacity=".55"/>
        <stop offset=".5" stop-color="{color}" stop-opacity="1"/>
        <stop offset="1" stop-color="{color}" stop-opacity=".55"/>
      </linearGradient></defs>
      <g fill="url(#wfg{uid})">{rects}</g>
      <style>
        .wf{uid} {{ transform-box: fill-box; transform-origin: center;
                    animation: wfGrow{uid} .55s cubic-bezier(.16,.84,.44,1) backwards; }}
        @keyframes wfGrow{uid} {{ from {{ transform: scaleY(.06); opacity: .25; }}
                                  to {{ transform: scaleY(1); opacity: 1; }} }}
      </style>
    </svg>
    """)


def friendly_time(stamp):
    try:
        t = datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return stamp
    today = now_local().date()
    if t.date() == today:
        day = "Today"
    elif t.date() == today - datetime.timedelta(days=1):
        day = "Yesterday"
    else:
        day = t.strftime("%d %b %Y")
    return f"{day}, {t:%H:%M}"


def model_short(model_label):
    return f"Model {model_label.split(' · ')[0]}" if " · " in model_label else model_label


def confidence_label(margin):
    if margin >= 0.7:
        return "Strong"
    if margin >= 0.35:
        return "Moderate"
    return "Borderline"


def _open_record(key):
    st.session_state.report_open = key
    st.session_state.force_scroll_top = True


def _close_record():
    st.session_state.report_open = None
    st.session_state.force_scroll_top = True


def _show_more_records():
    st.session_state.report_limit = st.session_state.get("report_limit", REPORT_PAGE_SIZE) + REPORT_PAGE_SIZE


def _verdict_style(label):
    is_gen = label == "Genuine"
    return (is_gen, C["genuine"] if is_gen else C["synthetic"],
            "Genuine" if is_gen else "Synthetic")


def render_clip_card(r):
    """One recording in the library grid: verdict, its real waveform, the
    recorded score, and when/how it was analysed — at a glance, before
    opening the full case file."""
    is_gen, color, chip = _verdict_style(r["prediction"])
    audio = record_audio(r)
    prof = clip_profile(audio, 44) if audio else None
    score = float(r["score"])
    wave = (waveform_svg(prof["bars"], color) if prof else
            f'<div class="clip-nowave">audio no longer stored</div>')
    duration = f"{prof['duration']:.1f}s" if prof else "—"
    name = html.escape(r["filename"])
    with st.container(border=True, key=f"clipcard_{'gen' if is_gen else 'syn'}_{r['id']}"):
        st.markdown(_flatten_markup(f"""
        <div class="clip-top">
          <span class="clip-chip" style="color:{color};background:{color}1a;border-color:{color}55;">
            {'●' if is_gen else '▲'} {chip}</span>
          <span class="clip-no">#{r['_no']:03d}</span>
        </div>
        <div class="clip-name" title="{name}">{name}</div>
        <div class="clip-wave">{wave}</div>
        <div class="clip-score"><span>Genuine score</span>
          <div class="bar"><i style="width:{score * 100:.0f}%;background:{color};"></i></div>
          <b style="color:{color}">{score * 100:.0f}%</b></div>
        <div class="clip-meta"><span>{friendly_time(r['analysed_at'])}</span>
          <span>{duration}</span><span>{model_short(r['model'])}</span></div>
        """), unsafe_allow_html=True)
        st.button("Open full report", key=f"open_{r['id']}", icon=":material/open_in_full:",
                  on_click=_open_record, args=(str(r["id"]),), use_container_width=True)


def render_library(records):
    newest_first = list(reversed(records))
    f1, f2, f3 = st.columns([1.25, 1.7, 1])
    with f1:
        verdict = st.segmented_control("Show", ["All", "Genuine", "Synthetic"], default="All",
                                       key="report_filter", label_visibility="collapsed")
    with f2:
        query = st.text_input("Search", placeholder="Search recordings by filename…",
                              key="report_search", label_visibility="collapsed",
                              icon=":material/search:")
    with f3:
        order = st.selectbox("Sort", ["Newest first", "Oldest first", "Most genuine", "Most synthetic"],
                             key="report_sort", label_visibility="collapsed")

    items = newest_first
    if verdict in ("Genuine", "Synthetic"):
        items = [r for r in items if r["prediction"] == verdict]
    if query:
        items = [r for r in items if query.strip().lower() in r["filename"].lower()]
    if order == "Oldest first":
        items = list(reversed(items))
    elif order == "Most genuine":
        items = sorted(items, key=lambda r: float(r["score"]), reverse=True)
    elif order == "Most synthetic":
        items = sorted(items, key=lambda r: float(r["score"]))

    if not items:
        st.info("No recordings match these filters.")
        return

    limit = st.session_state.get("report_limit", REPORT_PAGE_SIZE)
    shown = items[:limit]
    for start in range(0, len(shown), 3):
        cols = st.columns(3)
        for col, r in zip(cols, shown[start:start + 3]):
            with col:
                render_clip_card(r)
    if len(items) > limit:
        st.button(f"Show more recordings ({len(items) - limit} more)", icon=":material/expand_more:",
                  on_click=_show_more_records, use_container_width=True)


def render_case_file(r, newer, older):
    """The full report for one historical entry, laid out as a case file.

    Score, verdict and model are shown exactly as recorded at analysis time
    (never re-scored against whichever model happens to be selected today,
    so history stays an honest record of what actually happened). Playback
    and the signal-analysis plots are regenerated from the saved audio,
    which is the only part that doesn't need the original model to rebuild.
    """
    is_gen, color, chip = _verdict_style(r["prediction"])
    p_genuine, threshold, margin = float(r["score"]), float(r["threshold"]), _record_margin(r)
    audio = record_audio(r)
    loaded = load_audio_safely(audio) if audio else (None, None, "missing")
    y, sr, err = loaded
    prof = clip_profile(audio, 150) if audio and not err else None
    name = html.escape(r["filename"])
    confidence = confidence_label(margin)

    n1, _, n2, n3 = st.columns([1.5, 2.6, 0.9, 0.9])
    n1.button("Back to all recordings", icon=":material/arrow_back:", on_click=_close_record,
              use_container_width=True, key="case_back")
    n2.button("‹ Newer", disabled=newer is None, on_click=_open_record,
              args=(str(newer["id"]) if newer else "",), use_container_width=True, key="case_newer")
    n3.button("Older ›", disabled=older is None, on_click=_open_record,
              args=(str(older["id"]) if older else "",), use_container_width=True, key="case_older")

    backdrop = waveform_svg(prof["bars"], color, height=120) if prof else ""
    st.markdown(_flatten_markup(f"""
    <div class="case-head" style="border-color:{color}55;
         background:radial-gradient(120% 140% at 0% 0%, {color}26, transparent 55%),
                    linear-gradient(160deg, {C['panel2']}, {C['panel']});">
      <div class="case-backdrop">{backdrop}</div>
      <div class="case-main">
        <div class="case-kicker">Case file #{r['_no']:03d} &nbsp;·&nbsp; {friendly_time(r['analysed_at'])}</div>
        <div class="case-title" title="{name}">{name}</div>
        <span class="clip-chip" style="color:{color};background:{color}1a;border-color:{color}66;">
          {'● Genuine human voice' if is_gen else '▲ AI-generated / cloned voice'}</span>
      </div>
      <div class="case-score">
        <div class="v" style="color:{color};text-shadow:0 0 28px {color}66;">{p_genuine * 100:.0f}%</div>
        <div class="k">likelihood genuine</div>
        <div class="case-conf" style="border-color:{color}55;">{confidence} confidence</div>
      </div>
    </div>
    <div class="case-facts">
      <div><span>Analysed</span><b>{html.escape(r['analysed_at'])}</b></div>
      <div><span>Model</span><b>{html.escape(model_short(r['model']))}</b></div>
      <div><span>Threshold</span><b>{threshold:.2f}</b></div>
      <div><span>Margin</span><b>{margin * 100:.1f}%</b></div>
      <div><span>Duration</span><b>{f"{len(y) / sr:.2f}s" if not err else "—"}</b></div>
    </div>
    """), unsafe_allow_html=True)

    if err:
        st.warning("The original audio for this entry is no longer available, so playback and "
                   "signal plots can't be rebuilt — the recorded result above is unchanged.")
    else:
        st.markdown("#### Recording")
        with st.container(border=True, key="case_player"):
            st.markdown(f'<div class="case-wave">{waveform_svg(prof["bars"], color, height=72)}</div>',
                        unsafe_allow_html=True)
            st.audio(audio)
            d1, d2, _ = st.columns([1, 1, 2])
            d1.download_button("Download audio", audio, file_name=r["filename"],
                               icon=":material/download:", use_container_width=True, key="case_dl")
            with d2.popover("Delete recording", icon=":material/delete:", use_container_width=True):
                st.markdown("This permanently removes the clip and its report.")
                if st.button("Yes, delete it", type="primary", key="case_delete"):
                    history_delete(r["id"])
                    _close_record()
                    st.rerun()

    tabs = ["Verdict", "What this means"] + ([] if err else ["Waveform & spectrogram", "MFCC fingerprint"])
    t = st.tabs([f"{icon}  {label}" for icon, label in
                 zip(["🎯", "💡", "🌊", "🧬"], tabs)])
    with t[0]:
        left, right = st.columns([1, 1.6])
        with left:
            st.markdown(gauge_svg(p_genuine, r["prediction"], threshold), unsafe_allow_html=True)
        with right:
            result_card(r["filename"], r["prediction"], margin, len(y) / sr if not err else 0.0,
                        sr or SAMPLE_RATE, r["model"], threshold, p_genuine)
    with t[1]:
        if confidence == "Borderline":
            meaning = ("This score sits close to the decision threshold, so treat the verdict as "
                       "<b>inconclusive</b>. A longer, cleaner recording — or comparing models on the "
                       "Detect Voice page — will give a firmer answer.")
        elif is_gen:
            meaning = (f"The model found the patterns of a <b>real human voice</b> with {confidence.lower()} "
                       "confidence — natural variation in pitch, timing and breath that synthesis "
                       "engines tend to smooth over.")
        else:
            meaning = (f"The model found <b>signs of synthetic generation</b> with {confidence.lower()} "
                       "confidence — spectral regularities typical of text-to-speech and voice-cloning "
                       "engines. Treat requests made in this voice with caution and verify through "
                       "another channel.")
        note = MODEL_NOTES.get(available_models().get(r["model"]), "")
        st.markdown(f'<div class="card"><p style="font-size:14px;color:{C["text"]};">{meaning}</p></div>',
                    unsafe_allow_html=True)
        st.markdown(
            f"- **Score {p_genuine:.3f}** is the model's likelihood that the voice is genuine; anything "
            f"above the **{threshold:.2f}** threshold counts as Genuine.\n"
            f"- **Margin {margin * 100:.1f}%** is how far the score sits from that threshold, on its side "
            f"— higher means a more decisive verdict.\n"
            f"- **{r['model']}**" + (f": {note}" if note else "") + "\n"
            "- This is the result exactly as recorded at analysis time — it isn't re-scored if the "
            "models change later."
        )
    if not err:
        with t[2]:
            fig = plot_analysis(y, sr, r["filename"])
            st.pyplot(fig)
            plt.close(fig)
        with t[3]:
            fig2 = plot_mfcc(extract_mfcc(y, sr=sr))
            st.pyplot(fig2)
            plt.close(fig2)


def page_report():
    records = [dict(r, _no=i) for i, r in enumerate(history_records(), start=1)]
    newest_first = list(reversed(records))
    keys = [str(r["id"]) for r in newest_first]
    open_key = st.session_state.get("report_open")
    if open_key in keys:
        i = keys.index(open_key)
        render_case_file(newest_first[i],
                         newest_first[i - 1] if i > 0 else None,
                         newest_first[i + 1] if i + 1 < len(newest_first) else None)
        return

    if is_guest():
        hero("Evidence trail", "Session Report",
             "Every clip you've analysed in this visit — browse your recordings, replay them, and "
             "open any one for its full case file.")
        st.info("You're browsing as a guest, so this list is kept only until you leave — "
                "nothing is saved on the server. Sign up to keep a permanent history.")
    else:
        hero("Evidence trail", "Session Report",
             "Every clip you've ever analysed on this account — browse your recordings, replay them, "
             "and open any one for its full case file. Persists across logins.")

    if not records:
        st.markdown(_flatten_markup(f"""
        <div class="card" style="text-align:center;padding:42px 24px;">
          <div style="width:64px;margin:0 auto 14px auto;opacity:.8;">{mic_icon_svg()}</div>
          <h4>No recordings yet</h4>
          <p>Analyse a clip on the <b>Detect Voice</b> page and it will appear here, ready to replay
          and inspect.</p>
        </div>"""), unsafe_allow_html=True)
        return

    genuine = sum(1 for r in records if r["prediction"] == "Genuine")
    synthetic = len(records) - genuine
    stat_row([(str(len(records)), "Recordings"),
              (str(genuine), "Genuine"),
              (str(synthetic), "Synthetic"),
              (friendly_time(newest_first[0]["analysed_at"]).split(",")[0], "Last analysed")])
    st.markdown(_flatten_markup(f"""
    <div class="split-bar" title="{genuine} genuine · {synthetic} synthetic">
      <i style="width:{genuine / len(records) * 100:.1f}%;background:{C['genuine']};"></i>
      <i style="width:{synthetic / len(records) * 100:.1f}%;background:{C['synthetic']};"></i>
    </div>"""), unsafe_allow_html=True)

    st.markdown("### Your recordings")
    render_library(records)

    with st.expander("📋  Detailed log, export & clear"):
        cols = ["Filename", "Prediction", "Score", "Threshold", "Model", "Time"]
        rows = [{"Filename": r["filename"], "Prediction": r["prediction"], "Score": r["score"],
                 "Threshold": r["threshold"], "Model": r["model"], "Time": r["analysed_at"]}
                for r in records]
        st.table(rows)
        csv = ",".join(cols) + "\n" + "\n".join(
            ",".join(str(r.get(c, "")).replace(",", ";") for c in cols) for r in rows
        )
        c1, c2, _ = st.columns([1, 1, 2])
        c1.download_button("Export CSV", csv, file_name="detection_report.csv",
                           icon=":material/table_view:", use_container_width=True)
        with c2.popover("Clear my history", icon=":material/delete_sweep:", use_container_width=True):
            st.markdown("This permanently deletes **every** recording and report on this "
                        + ("visit." if is_guest() else "account."))
            if st.button("Yes, clear everything", type="primary", key="clear_history_confirm"):
                history_clear()
                st.rerun()


def page_coverage():
    hero("Scope & honesty", "Detection Coverage",
         "Every real detector has a boundary. This page states it explicitly: what's "
         "measured and working, what's measured and failing, and what's genuinely "
         "still unknown — rather than a single blanket claim of \"detects synthetic voices.\"")

    st.markdown("### ✅ Confirmed — detects well (measured)")
    status_card_row([
        ("🎯", "ASVspoof 2019 attacks (seen)",
         "2019-era neural TTS + voice conversion, the six attack types used in training. "
         "97.4-97.6% accuracy, 5.83-8.47% EER."),
        ("🌍", "Modern real-world neural clones",
         "In-the-Wild-style deepfakes. Directly trained on this data — 95.6-96.5% "
         "real-world accuracy, up from 46.8% before it was added."),
        ("🕹️", "Classical / formant TTS — model 4 only",
         "Windows SAPI, espeak-ng. Fixed in the recommended model (4 · Combined + classical "
         "TTS): 3/3 correct on held-out test clips, up from 1/3 in model 3, by adding 120 "
         "such clips to training."),
        ("📋", "10 of 13 unseen ASVspoof attacks",
         "Tested against the full 71,237-clip eval set — 13 attack types absent from all "
         "training data. 10 of 13 (TTS and hybrid TTS/VC methods) caught at 85-100%, "
         "consistently across every model version — genuine generalisation, not memorisation."),
    ], C["genuine"])

    st.markdown("### ❌ Confirmed — escapes detection (measured, failing)")
    status_card_row([
        ("🕹️", "Classical / formant TTS — models 2 and 3",
         "The fine-tuned and combined models (before the classical-TTS fix) still miss most "
         "of these — 0/3 and 1/3 correct respectively. Model 1 (baseline) also flags them, "
         "but likely only because it mislabels almost anything unfamiliar as synthetic (see "
         "its 23% real-world recall), not genuine detection skill — so it isn't a model worth "
         "crediting for this either."),
        ("🧬", "Pure voice conversion (A17, A18, A19)",
         "The 3 remaining unseen ASVspoof attacks — officially classified as pure "
         "signal-processing voice conversion (waveform filtering, vocoder, spectral "
         "filtering), not neural generation — are caught only 17-35% of the time, "
         "consistently across all four model versions. This is the project's core target "
         "scenario: cloning a real person while preserving their natural prosody. Actively "
         "being addressed with additional voice-conversion training data (see About)."),
    ], C["synthetic"])

    st.markdown("### ❓ Unknown — never actually tested")
    status_card_row([
        ("✨", "Modern commercial cloning tools (2023+)",
         "Both training datasets predate today's most advanced diffusion/flow-based "
         "cloning engines. No reference for their artifacts exists in training."),
        ("📞", "Telephone-codec compressed audio",
         "8kHz G.711/GSM — the actual medium of real scam calls. Never trained or "
         "tested on codec-degraded audio at all."),
    ], C["warn"])

    st.markdown(
        f"""<div class="banner"><b>Why state this at all.</b> A demo that says exactly what's
        proven, what's disproven, and what's still an open question reads as more credible
        than a claim of universal detection — and it's the honest position: no static
        detector, from any team, can claim to catch every synthesis method that exists or
        will exist. This is an evolving arms race, not a solved problem. The one confirmed
        weak spot — pure voice conversion — is precisely scoped and has an active plan to
        close it, rather than being an unexamined gap.</div>""",
        unsafe_allow_html=True,
    )


def page_about():
    hero("Project information", "About",
         "Scope, technology stack and planned extensions for the Synthetic Voice Detection System.")

    st.markdown("### Technology stack")
    card_row([
        ("🧠", "Deep learning", "PyTorch — CNN + bidirectional LSTM trained on a Google Colab T4 GPU."),
        ("🎵", "Audio processing", "Librosa and Torchaudio for loading, resampling and MFCC extraction."),
        ("🖥️", "Interface", "Streamlit, serving the interactive web application you are using now."),
    ])

    st.markdown("### Datasets")
    st.table({
        "Dataset": ["ASVspoof 2019 (Logical Access)", "In-the-Wild deepfake dataset"],
        "Role": ["Primary training and evaluation", "Real-world robustness fine-tuning"],
        "Character": ["Clean studio recordings, known attack types",
                      "Genuine and cloned speech from real-world sources"],
    })

    st.markdown("### Applications")
    card_row([
        ("🏦", "Financial fraud", "Flag cloned-voice impersonation in phone-based authorisation attempts."),
        ("⚖️", "Forensic analysis", "Provide a documented, replayable assessment of disputed audio evidence."),
        ("📱", "Media verification", "Screen circulating audio clips before they are treated as authentic."),
    ])

    st.markdown("### Planned extensions")
    steps([
        ("Close the pure voice-conversion gap",
         "The system's one confirmed weak spot: 3 of 13 unseen ASVspoof attack types (A17-A19, pure "
         "signal-processing voice conversion) are caught only 17-35% of the time, versus 85-100% for "
         "everything else, consistently across every model version. Plan: add real voice-conversion "
         "training examples — from the VCC2018 Voice Conversion Challenge dataset and/or generated "
         "locally via WORLD-vocoder spectral-envelope warping — as a new training bucket, using the "
         "same recipe that already fixed the classical-TTS gap."),
        ("Integrated voice cloning demonstration",
         "Add an open-source text-to-speech module so that generation and detection can be demonstrated "
         "side by side within a single self-contained application."),
        ("Streaming detection",
         "Extend from clip-level analysis to continuous monitoring of a live call."),
    ])


def password_field(label, key, help=None):
    """A text_input with a per-field show/hide toggle, since Streamlit has
    no built-in one — the checkbox's own key is derived from `key` so two
    password fields on the same form (e.g. password + confirm) don't fight
    over the same show/hide state."""
    show = st.checkbox("Show password", key=f"{key}_show", value=False)
    return st.text_input(label, type="default" if show else "password", key=key, help=help)


def inject_auth_theme():
    """A background unique to the login screen — concentric rings pulsing
    outward from center, like a mic listening for a voice, layered over the
    same cursor-reactive dust field the rest of the app uses (inject_particles
    is called with echo=True right after this, so the dust also reacts to
    each ring). Also hides the sidebar chrome entirely, since it's empty
    until a user logs in.

    The gradient is applied to `.stApp`, not `[data-testid="stAppViewContainer"]`
    — inject_particles forces that element's background to transparent (with
    !important, so its canvas shows through) regardless of injection order,
    which would silently erase a gradient placed there instead.
    """
    st.markdown(
        f"""
        <style>
        [data-testid="collapsedControl"] {{ display: none; }}
        .stApp {{
            background:
                radial-gradient(circle at 50% 42%, {C['panel2']}55 0%, transparent 60%),
                radial-gradient(ellipse at center, {C['bg']} 0%, #05090c 100%);
            overflow: hidden;
        }}
        .auth-ring {{
            position: fixed; top: 42%; left: 50%; border-radius: 50%;
            border: 1px solid {C['accent']}66;
            transform: translate(-50%, -50%);
            width: 40px; height: 40px; opacity: 0;
            animation: authPulse 4.5s cubic-bezier(.15,.65,.3,1) infinite;
            pointer-events: none; z-index: 0;
        }}
        @keyframes authPulse {{
            0% {{ width: 40px; height: 40px; opacity: .85; }}
            80% {{ opacity: .12; }}
            100% {{ width: 1500px; height: 1500px; opacity: 0; }}
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {C['panel']}cc !important;
            backdrop-filter: blur(14px);
            border: 1px solid {C['accent']}40 !important;
            border-radius: 22px !important;
            box-shadow: 0 20px 60px #00000066, 0 0 0 1px {C['border']}80;
            padding: 6px 8px;
        }}
        </style>
        <div class="auth-ring" style="animation-delay:0s;"></div>
        <div class="auth-ring" style="animation-delay:1.5s;"></div>
        <div class="auth-ring" style="animation-delay:3s;"></div>
        """,
        unsafe_allow_html=True,
    )


def page_auth():
    inject_auth_theme()
    inject_particles(echo=True)
    st.markdown("<div style='height:5vh'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.15, 1])
    with mid:
        st.markdown(
            f"""<div style="text-align:center;margin-bottom:18px;position:relative;z-index:1;">
              <div style="width:64px;height:64px;margin:0 auto 14px auto;border-radius:50%;
                   background:linear-gradient(135deg,{C['accent']},{C['accent2']});
                   display:flex;align-items:center;justify-content:center;
                   box-shadow:0 0 32px {C['accent']}55;">{mic_icon_svg()}</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:24px;
                   color:{C['text']};">Synthetic Voice Detection</div>
              <div style="color:{C['muted']};font-size:13px;margin-top:4px;">
                   Sign in, or try it instantly as a guest</div>
            </div>""",
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            mode = option_menu(
                menu_title=None,
                options=["Sign In", "Sign Up", "Forgot Password"],
                icons=["box-arrow-in-right", "person-plus", "key"],
                orientation="horizontal",
                default_index=["signin", "signup", "forgot"].index(st.session_state.auth_mode),
                key=f"auth_mode_menu_{st.session_state.auth_nonce}",
                styles={
                    "container": {"padding": "5px", "background-color": C["panel2"],
                                  "border": f"1px solid {C['border']}", "border-radius": "14px"},
                    "icon": {"color": C["accent"], "font-size": "14px"},
                    "nav-link": {"font-size": "12px", "text-align": "center", "padding": "10px 6px",
                                 "border-radius": "10px", "color": C["muted"], "margin": "0 2px", "font-weight": "600"},
                    "nav-link-selected": {"background": f"linear-gradient(135deg,{C['accent']},{C['accent2']})",
                                           "color": C["bg"], "font-weight": "700"},
                },
            )
            new_mode = {"Sign In": "signin", "Sign Up": "signup", "Forgot Password": "forgot"}[mode]
            if new_mode != st.session_state.auth_mode:
                st.session_state.auth_mode = new_mode
                st.session_state.forgot_stage = 0
                st.rerun()

            if st.session_state.auth_mode == "signin":
                st.markdown("#### Sign In")
                username = st.text_input("Username", key="signin_username")
                password = password_field("Password", key="signin_password")
                if st.button("Sign In", type="primary", use_container_width=True):
                    if not username or not password:
                        st.error("Enter both your username and password.")
                    else:
                        user = auth.verify_login(username, password)
                        if user is None:
                            st.error("Incorrect username or password.")
                        else:
                            st.session_state.authenticated = True
                            st.session_state.user = user
                            st.rerun()

            elif st.session_state.auth_mode == "signup":
                st.markdown("#### Sign Up")
                col1, col2 = st.columns(2)
                with col1:
                    first_name = st.text_input("First name", key="signup_first")
                with col2:
                    last_name = st.text_input("Last name", key="signup_last")
                username = st.text_input("Username", key="signup_username")
                phone_number = st.text_input("Mobile number (optional)", key="signup_phone")
                password = password_field(
                    "Password", key="signup_password",
                    help="At least 8 characters, with uppercase, lowercase, and a number.",
                )
                if password:
                    label, tone = auth.password_strength_label(password)
                    st.markdown(f"""<div style="color:{C[tone]};font-size:12.5px;margin:-8px 0 8px 2px;">
                        Password strength: <b>{label}</b></div>""", unsafe_allow_html=True)
                confirm_password = password_field("Confirm password", key="signup_confirm")
                security_question = st.selectbox("Security question (for password reset)",
                                                 auth.SECURITY_QUESTIONS, key="signup_question")
                security_answer = st.text_input("Answer", key="signup_answer")

                if st.button("Create Account", type="primary", use_container_width=True):
                    valid, msg = auth.validate_password_strength(password)
                    if not first_name or not last_name or not username or not password:
                        st.error("First name, last name, username, and password are all required.")
                    elif not valid:
                        st.error(msg)
                    elif password != confirm_password:
                        st.error("Passwords do not match.")
                    elif not security_answer:
                        st.error("Please answer the security question — it's needed to reset your password later.")
                    elif auth.username_exists(username):
                        st.error(f"Username '{username}' is already taken.")
                    else:
                        auth.create_user(first_name, last_name, username, password,
                                         phone_number, security_question, security_answer)
                        st.success("Account created — you can now sign in.")
                        st.session_state.auth_mode = "signin"
                        st.session_state.auth_nonce += 1
                        st.rerun()

            else:  # forgot
                st.markdown("#### Forgot Password")
                if st.session_state.forgot_stage == 0:
                    username = st.text_input("Your username", key="forgot_username_input")
                    if st.button("Continue", type="primary", use_container_width=True):
                        question = auth.get_security_question(username)
                        if question is None:
                            st.error("No account found with that username.")
                        else:
                            st.session_state.forgot_username = username
                            st.session_state.forgot_question = question
                            st.session_state.forgot_stage = 1
                            st.rerun()
                elif st.session_state.forgot_stage == 1:
                    st.caption(f"Signed in as **{st.session_state.forgot_username}**")
                    st.markdown(f"**{st.session_state.forgot_question}**")
                    answer = st.text_input("Your answer", key="forgot_answer_input")
                    if st.button("Verify", type="primary", use_container_width=True):
                        if auth.verify_security_answer(st.session_state.forgot_username, answer):
                            st.session_state.forgot_stage = 2
                            st.rerun()
                        else:
                            st.error("That answer doesn't match our records.")
                else:
                    st.caption(f"Resetting password for **{st.session_state.forgot_username}**")
                    new_password = password_field(
                        "New password", key="forgot_new_password",
                        help="At least 8 characters, with uppercase, lowercase, and a number.",
                    )
                    confirm = password_field("Confirm new password", key="forgot_confirm_password")
                    if st.button("Reset Password", type="primary", use_container_width=True):
                        valid, msg = auth.validate_password_strength(new_password)
                        if not valid:
                            st.error(msg)
                        elif new_password != confirm:
                            st.error("Passwords do not match.")
                        else:
                            auth.reset_password(st.session_state.forgot_username, new_password)
                            st.success("Password reset — you can now sign in with your new password.")
                            st.session_state.auth_mode = "signin"
                            st.session_state.auth_nonce += 1
                            st.session_state.forgot_stage = 0
                            st.rerun()

        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:12px;margin:14px 0 10px 0;
                 color:{C['muted']};font-size:12px;position:relative;z-index:1;">
              <div style="flex:1;height:1px;background:{C['border']};"></div>or
              <div style="flex:1;height:1px;background:{C['border']};"></div></div>""",
            unsafe_allow_html=True,
        )
        if st.button("Continue as guest", icon=":material/person:", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user = {"id": None, "first_name": "Guest", "is_guest": True}
            st.session_state.guest_records = []
            st.rerun()
        st.caption("Guests can use every feature, but nothing they analyse is saved. "
                   "Signed-in users' clips are stored so they can be reopened later. "
                   "Accounts on the public demo may be reset from time to time.")


# ============================================================
# APP
# ============================================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Read before set_page_config (the one Streamlit call that must run first)
# so the sidebar starts fully collapsed on the login screen — there's
# nothing in it until a user is signed in.
st.set_page_config(
    page_title="Synthetic Voice Detection", page_icon="🎙️", layout="wide",
    initial_sidebar_state="expanded" if st.session_state.authenticated else "collapsed",
)

auth.init_db()

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
# Swap the shared palette dict IN PLACE (not reassigned) before anything
# below reads C['...'] — every function already does that lookup at call
# time, so this one line is the whole theme switch; nothing else changes.
C.clear()
C.update(C_DARK if st.session_state.dark_mode else C_LIGHT)

inject_css()
inject_micro_interactions()

if "user" not in st.session_state:
    st.session_state.user = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "signin"
if "auth_nonce" not in st.session_state:
    st.session_state.auth_nonce = 0
if "forgot_stage" not in st.session_state:
    st.session_state.forgot_stage = 0

if not st.session_state.authenticated:
    page_auth()
    st.stop()

inject_particles()

if "threshold" not in st.session_state:
    st.session_state.threshold = DEFAULT_THRESHOLD
if "thr_nonce" not in st.session_state:
    st.session_state.thr_nonce = 0
if "pending_recording" not in st.session_state:
    st.session_state.pending_recording = None
if "recording_analysed" not in st.session_state:
    st.session_state.recording_analysed = False
if "rec_nonce" not in st.session_state:
    st.session_state.rec_nonce = 0
if "upload_sig" not in st.session_state:
    st.session_state.upload_sig = None
if "upload_analysed" not in st.session_state:
    st.session_state.upload_analysed = False

models = available_models()

with st.sidebar:
    st.markdown(
        f"""<div style="padding:4px 2px 16px 2px;">
          <div style="display:flex;align-items:center;gap:9px;">
            <div style="width:9px;height:9px;border-radius:50%;background:{C['genuine']};
                 box-shadow:0 0 12px {C['genuine']};"></div>
            <div class="brand-title">VoiceGuard</div>
          </div>
          <div style="color:{C['muted']};font-family:'JetBrains Mono',monospace;
               font-size:9.5px;letter-spacing:1.2px;text-transform:uppercase;margin-top:5px;">
          Synthetic Voice Detection</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div style="color:{C['text']};font-size:12px;margin:-6px 0 12px 2px;">
        Welcome, <b>{st.session_state.user['first_name']}</b></div>""",
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title=None,
        options=["Overview", "Detect Voice", "How It Works", "Model & Results",
                 "Coverage", "Session Report", "About"],
        icons=["grid-1x2", "soundwave", "diagram-3", "graph-up",
               "shield-check", "clipboard-data", "info-circle"],
        default_index=0,
        styles={
            # This component renders inside its OWN embedded iframe, with its own
            # default (white) page background — "transparent" here reveals THAT,
            # not our dark sidebar behind it. Must be an explicit dark fill.
            "container": {"padding": "0", "background-color": C["panel"]},
            "icon": {"color": C["accent"], "font-size": "14px"},
            "nav-link": {
                "font-size": "13px", "text-align": "left", "margin": "2px 0",
                "padding": "10px 12px", "border-radius": "8px", "color": C["muted"],
                "background-color": C["panel"], "--hover-color": C["panel2"],
            },
            "nav-link-selected": {
                "background-color": C["accent"] + "26", "color": C["text"],
                "font-weight": "600", "border-left": f"2px solid {C['accent']}",
                "border-radius": "8px",
            },
        },
    )

    st.markdown("---")
    # Filled in after the page renders (bottom of this file), so ANALYSED
    # already includes any clip analysed during this same rerun.
    status_slot = st.empty()

    dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode, key="dark_mode_toggle")
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()

    if st.button("Sign in / Sign up" if is_guest() else "Log out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.guest_records = []
        st.rerun()

# Only reset scroll when the page actually changed — not on every rerun,
# or moving the threshold slider etc. would keep yanking the view to the top.
# Checked here (before the page renders) so the flag update itself doesn't
# depend on page content, but the actual scroll_to_top() call happens only
# after that page has fully rendered (see below the footer) — resetting
# scroll before the new content exists lets the browser's scroll-anchoring
# quietly pull it back down as that content streams in afterward.
if "last_page" not in st.session_state:
    st.session_state.last_page = selected
page_changed = st.session_state.last_page != selected
st.session_state.last_page = selected
if page_changed:
    st.session_state.report_open = None  # Session Report reopens on the library, not a stale case

if selected == "Overview":
    page_overview(models)
elif selected == "Detect Voice":
    page_detect(models)
elif selected == "How It Works":
    page_how()
elif selected == "Model & Results":
    page_model(models)
elif selected == "Coverage":
    page_coverage()
elif selected == "Session Report":
    page_report()
else:
    page_about()

st.markdown(
    f"""<div style="text-align:center;color:{C['muted']};font-size:11.5px;
    padding:26px 0 10px 0;border-top:1px solid {C['border']};margin-top:34px;">
    Synthetic Voice Detection System &nbsp;•&nbsp; CNN-LSTM &nbsp;•&nbsp; ASVspoof 2019 LA + In-the-Wild
    </div>""",
    unsafe_allow_html=True,
)

status_slot.markdown(
    f"""<div style="font-family:'JetBrains Mono',monospace;font-size:9.5px;
         color:{C['muted']};letter-spacing:1.3px;text-transform:uppercase;
         margin-bottom:10px;">System status</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:11.5px;
         color:{C['muted']};line-height:2.1;">
      <div style="display:flex;justify-content:space-between;">
        <span>COMPUTE</span><b style="color:{C['accent']}">{device.upper()}</b></div>
      <div style="display:flex;justify-content:space-between;">
        <span>MODELS</span><b style="color:{C['accent']}">{len(models)}</b></div>
      <div style="display:flex;justify-content:space-between;">
        <span>ANALYSED</span><b style="color:{C['accent']}">{history_count()}</b></div>
    </div>""",
    unsafe_allow_html=True,
)

if page_changed or st.session_state.pop("force_scroll_top", False):
    scroll_to_top()
