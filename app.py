import html
import os
import tempfile
import time

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
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
                        "eer": None},
}

# Score at which false acceptances and false rejections are equal, on ASVspoof dev.
# No entry for a model means EER hasn't been measured for it yet.
EER_THRESHOLDS = {
    BASELINE_MODEL: 0.0074,
    FINETUNED_MODEL: 0.8572,
    COMBINED_MODEL: 0.0042,
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

# "Evidence Console" design tokens — a forensic instrument, not an AI-demo
# showcase. Color carries meaning only: blue = interaction/focus, green =
# genuine, red = synthetic, amber = warning, everything else is neutral ink.
# No decorative gradients, no glow-as-default — restraint is the point.
C = {
    "bg": "#0b0d10",
    "panel": "#14171c",
    "panel2": "#1b1f26",
    "border": "#262b33",
    "text": "#f4f5f7",
    "muted": "#8b92a0",
    "accent": "#4f7fff",
    "genuine": "#16a34a",
    "synthetic": "#dc2626",
    "warn": "#d97706",
}

# Type scale — one assigned job per size, replacing the ad-hoc 11/11.5/12/12.5px
# values accumulated across many earlier edits.
TYPE = {
    "xs": "12px", "sm": "13px", "body": "15px", "lg": "17px",
    "h4": "20px", "h3": "26px", "h2": "34px", "h1": "44px",
}

# Spacing — strict 4px base unit; every padding/margin in the app draws from
# this set rather than being hand-picked per component.
SPACE = {"1": "4px", "2": "8px", "3": "12px", "4": "16px", "5": "24px", "6": "32px", "7": "48px"}

# Elevation — layered shadows so panels read as surfaces, not flat outlines.
# Glow is reserved for a handful of high-signal elements (primary button,
# active nav item, the verdict card) rather than applied everywhere.
SHADOW = {
    "sm": "0 1px 3px rgba(0,0,0,.4)",
    "md": "0 10px 28px -12px rgba(0,0,0,.55)",
    "lg": "0 22px 48px -16px rgba(0,0,0,.6)",
}


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
    return y, sr, None


# ============================================================
# UI HELPERS
# ============================================================
def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

        html, body, .stApp, [class*="st-emotion"] {{
            font-family: 'IBM Plex Sans', -apple-system, sans-serif;
        }}
        /* Streamlit's Material icons are ligature fonts — the rule above would
           otherwise render them as their literal names ("upload", "arrow_right"). */
        [data-testid="stIconMaterial"], .material-symbols-rounded,
        .material-icons, [class*="material-symbols"] {{
            font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        }}
        .stApp {{
            background:
                radial-gradient(1100px 560px at 12% -12%, {C['accent']}0c 0%, transparent 58%),
                radial-gradient(900px 520px at 100% 0%, {C['accent']}07 0%, transparent 55%),
                radial-gradient(800px 480px at 50% 112%, {C['genuine']}05 0%, transparent 60%),
                repeating-linear-gradient(0deg, {C['accent']}05 0 1px, transparent 1px 72px),
                repeating-linear-gradient(90deg, {C['accent']}05 0 1px, transparent 1px 72px),
                {C['bg']};
            color: {C['text']};
        }}
        section[data-testid="stSidebar"] {{
            background: {C['panel']};
            border-right: 1px solid {C['border']};
        }}
        #MainMenu, footer {{ visibility: hidden; }}

        /* ---------- typography ---------- */
        h3 {{
            font-size: {TYPE['h4']} !important; font-weight: 600 !important;
            letter-spacing: -.2px; margin: {SPACE['6']} 0 {SPACE['3']} 0 !important;
        }}
        h3::before {{
            content: ""; display: inline-block; width: 3px; height: .85em;
            background: {C['accent']}; margin-right: {SPACE['2']}; vertical-align: -2px;
            border-radius: 1px;
        }}
        code {{ color: {C['accent']} !important; font-family: 'IBM Plex Mono', monospace !important; }}
        table {{ color: {C['text']} !important; font-size: {TYPE['sm']}; border-collapse: separate !important; }}
        thead th {{
            color: {C['muted']} !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: {TYPE['xs']} !important; letter-spacing: .5px; text-transform: uppercase;
            background: {C['panel2']} !important;
        }}
        tbody tr:hover td {{ background: {C['panel2']} !important; }}

        /* ---------- hero: composed page header, not a showcase banner ---------- */
        .hero {{
            padding: {SPACE['6']} {SPACE['6']};
            border-radius: 12px;
            background: radial-gradient(640px 220px at 88% -30%, {C['accent']}12, transparent 62%), {C['panel']};
            border: 1px solid {C['border']};
            margin-bottom: {SPACE['4']};
            animation: riseIn .35s ease-out;
            position: relative; overflow: hidden;
            box-shadow: {SHADOW['md']};
        }}
        .hero::before, .hero::after {{
            content: ""; position: absolute; width: 22px; height: 22px;
            opacity: .5; pointer-events: none;
        }}
        .hero::before {{
            top: 10px; left: 10px;
            border-top: 1.5px solid {C['accent']}; border-left: 1.5px solid {C['accent']};
            border-radius: 4px 0 0 0;
        }}
        .hero::after {{
            bottom: 10px; right: 10px;
            border-bottom: 1.5px solid {C['accent']}; border-right: 1.5px solid {C['accent']};
            border-radius: 0 0 4px 0;
        }}
        @keyframes riseIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        .hero-row {{
            display: flex; justify-content: space-between; align-items: flex-start;
            gap: {SPACE['6']}; flex-wrap: wrap; position: relative; z-index: 1;
        }}
        .hero-main {{ flex: 1 1 380px; min-width: 0; }}
        .hero-meta {{ display: flex; flex-direction: column; gap: {SPACE['3']}; padding-top: {SPACE['1']}; }}
        .hero-meta-item {{ text-align: right; }}
        .hero-meta-item .k {{
            display: block; font-family: 'IBM Plex Mono', monospace; font-size: {TYPE['xs']};
            color: {C['muted']}; letter-spacing: 1px; text-transform: uppercase;
        }}
        .hero-meta-item .v {{
            display: block; font-family: 'IBM Plex Mono', monospace; font-size: {TYPE['lg']};
            color: {C['text']}; font-weight: 600; margin-top: 2px;
        }}
        .pill {{
            display: inline-block; padding: 3px 10px; border-radius: 4px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: {TYPE['xs']}; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
            background: {C['panel2']}; color: {C['accent']};
            border: 1px solid {C['border']}; margin-bottom: {SPACE['3']};
        }}
        .hero h1 {{
            margin: 0; font-size: {TYPE['h1']}; font-weight: 700; letter-spacing: -.8px;
            color: {C['text']};
        }}
        .hero p {{
            color: {C['muted']}; margin-top: {SPACE['2']}; font-size: {TYPE['body']}; max-width: 680px;
            line-height: 1.6;
        }}

        /* ---------- unified card primitive ---------- */
        .ec-card {{
            background: {C['panel']};
            border: 1px solid {C['border']};
            border-left: 3px solid {C['border']};
            border-radius: 8px; padding: {SPACE['4']} {SPACE['5']}; height: 100%;
            box-shadow: {SHADOW['sm']};
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }}
        .ec-card:hover {{ box-shadow: {SHADOW['md']}; }}
        .ec-card--good {{ border-left-color: {C['genuine']}; }}
        .ec-card--bad {{ border-left-color: {C['synthetic']}; }}
        .ec-card--warn {{ border-left-color: {C['warn']}; }}
        .ec-card--interactive:hover {{
            border-color: {C['accent']}; transform: translateY(-3px);
            box-shadow: {SHADOW['lg']}, 0 0 0 1px {C['accent']}22;
        }}
        .ec-card h4 {{ margin: 0 0 {SPACE['1']} 0; font-size: {TYPE['body']}; color: {C['text']}; font-weight: 600; }}
        .ec-card p {{ margin: 0; font-size: {TYPE['sm']}; color: {C['muted']}; line-height: 1.6; }}
        .ec-card .ico {{ font-size: 20px; display: block; margin-bottom: {SPACE['3']}; }}
        .ec-chip {{
            display: inline-block; padding: 3px 9px; border-radius: 4px;
            background: {C['panel2']}; border: 1px solid {C['border']}; color: {C['muted']};
            font-family: 'IBM Plex Mono', monospace; font-size: {TYPE['xs']};
        }}
        @keyframes resultReveal {{
            from {{ opacity: 0; transform: translateY(8px) scale(.98); }}
            to   {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
        .result-reveal {{ animation: resultReveal .4s cubic-bezier(.16,.84,.44,1); }}

        /* ---------- stats ---------- */
        .stat {{
            background: {C['panel']};
            border: 1px solid {C['border']};
            border-radius: 8px; padding: {SPACE['4']} {SPACE['3']}; text-align: center;
            min-height: 96px;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            box-shadow: {SHADOW['sm']};
            transition: box-shadow .18s ease, border-color .18s ease;
        }}
        .stat:hover {{ box-shadow: {SHADOW['md']}; border-color: {C['accent']}44; }}
        .stat .v {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: {TYPE['h4']}; font-weight: 600; color: {C['text']}; line-height: 1.2;
        }}
        .stat .k {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: {TYPE['xs']}; color: {C['muted']}; text-transform: uppercase;
            letter-spacing: 1px; margin-top: {SPACE['2']};
        }}

        /* ---------- numbered steps ---------- */
        .step {{
            display: flex; gap: {SPACE['4']}; align-items: flex-start;
            background: {C['panel']}; border: 1px solid {C['border']};
            border-radius: 8px; padding: {SPACE['4']}; margin-bottom: {SPACE['2']};
            box-shadow: {SHADOW['sm']};
        }}
        .step .n {{
            flex: 0 0 28px; height: 28px; border-radius: 6px;
            background: {C['panel2']}; border: 1px solid {C['border']};
            color: {C['accent']}; font-family: 'IBM Plex Mono', monospace;
            font-weight: 600; display: flex;
            align-items: center; justify-content: center; font-size: {TYPE['sm']};
        }}
        .step h5 {{ margin: 2px 0 {SPACE['1']} 0; font-size: {TYPE['body']}; color: {C['text']}; font-weight: 600; }}
        .step p {{ margin: 0; font-size: {TYPE['sm']}; color: {C['muted']}; line-height: 1.6; }}

        /* ---------- inline notice banner ---------- */
        .banner {{
            border-radius: 6px; padding: {SPACE['3']} {SPACE['4']}; margin: {SPACE['2']} 0 {SPACE['4']} 0;
            border-left: 3px solid {C['warn']};
            background: {C['panel']};
            color: {C['text']}; font-size: {TYPE['sm']}; line-height: 1.6;
        }}
        .banner b {{ color: {C['warn']}; }}

        /* ---------- status strip (Detect Voice capture state) ---------- */
        .status-strip {{
            display: flex; align-items: center; gap: {SPACE['2']};
            padding: {SPACE['2']} {SPACE['1']} {SPACE['3']} {SPACE['1']};
        }}
        .status-strip .dot {{
            width: 7px; height: 7px; border-radius: 50%; flex: 0 0 auto;
        }}
        .status-strip .label {{
            font-family: 'IBM Plex Mono', monospace; font-size: {TYPE['xs']};
            letter-spacing: 1px; color: {C['muted']}; text-transform: uppercase;
        }}
        .spec-line {{
            font-family: 'IBM Plex Mono', monospace; font-size: {TYPE['xs']};
            color: {C['muted']}; letter-spacing: .3px; margin: 0 0 {SPACE['3']} 0;
        }}

        /* ---------- streamlit native controls ---------- */
        div[data-testid="stFileUploaderDropzone"] {{
            background: {C['panel']};
            border: 1.5px dashed {C['border']};
            border-radius: 8px;
            transition: border-color .15s ease;
        }}
        div[data-testid="stFileUploaderDropzone"]:hover {{ border-color: {C['accent']}; }}

        div[data-testid="stAlert"] {{
            border-radius: 8px !important; font-size: {TYPE['sm']} !important;
        }}

        .stTabs [data-baseweb="tab"] {{ color: {C['muted']}; font-size: {TYPE['sm']}; }}
        .stTabs [aria-selected="true"] {{ color: {C['accent']} !important; }}
        .stTabs [data-baseweb="tab-highlight"] {{ background: {C['accent']} !important; }}

        /* ---------- bordered containers (input console panels) ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid {C['border']} !important;
            border-radius: 10px !important;
            background: {C['panel']};
            box-shadow: {SHADOW['md']};
        }}

        /* ---------- capture affordances (Detect Voice) — static, not a fake meter ---------- */
        .io-glyph {{
            width: 52px; height: 52px; border-radius: 10px; margin: 0 auto {SPACE['3']} auto;
            display: flex; align-items: center; justify-content: center; font-size: 22px;
            background: {C['panel2']}; border: 1px solid {C['border']};
        }}
        .io-caption {{ text-align: center; padding: {SPACE['2']} 0 {SPACE['4']} 0; }}
        .io-caption .t {{ color: {C['text']}; font-weight: 600; font-size: {TYPE['body']}; }}
        .io-caption .s {{ color: {C['muted']}; font-size: {TYPE['xs']}; margin-top: {SPACE['1']}; }}

        /* ---------- buttons ---------- */
        div[data-testid="stButton"] button {{
            border-radius: 6px !important; font-weight: 600 !important;
            border: 1px solid {C['border']} !important;
            transition: border-color .15s ease, transform .15s ease, box-shadow .15s ease, background-color .15s ease;
        }}
        div[data-testid="stButton"] button:hover {{
            border-color: {C['accent']} !important; transform: translateY(-1px);
            background-color: {C['panel2']} !important;
        }}
        div[data-testid="stButton"] button[kind="primary"] {{
            background: {C['accent']} !important;
            border: none !important; color: #ffffff !important;
            box-shadow: 0 8px 20px -10px {C['accent']}80;
        }}
        div[data-testid="stButton"] button[kind="primary"]:hover {{
            box-shadow: 0 12px 28px -10px {C['accent']}99; transform: translateY(-1px);
        }}
        div[data-testid="stButton"] button:focus-visible,
        input:focus-visible, textarea:focus-visible, [tabindex]:focus-visible {{
            outline: 2px solid {C['accent']} !important; outline-offset: 2px;
        }}

        /* ---------- mobile ---------- */
        @media (max-width: 768px) {{
            .hero {{ padding: {SPACE['4']}; }}
            .hero h1 {{ font-size: {TYPE['h2']}; }}
            .hero-meta-item {{ text-align: left; }}
            .ec-card, .stat, .step {{ padding: {SPACE['3']}; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(pill, title, subtitle, meta=None):
    """`meta` is an optional list of (label, value) pairs — real system/session
    data (models loaded, session count, etc.), not decorative filler — shown
    as a small technical readout to the right of the title."""
    meta_html = ""
    if meta:
        items = "".join(
            f"""<div class="hero-meta-item"><span class="k">{html.escape(str(k))}</span>
                <span class="v">{html.escape(str(v))}</span></div>"""
            for k, v in meta
        )
        meta_html = f"""<div class="hero-meta">{items}</div>"""
    # No line in this block may be blank/whitespace-only: Markdown treats that
    # as ending the raw-HTML block early, which leaks the remaining closing
    # tags out as a literal code block. Keeping every line non-empty (the
    # {meta_html} placeholder sits on the same line as its neighbours rather
    # than alone) avoids that regardless of whether meta is supplied.
    st.markdown(
        f"""<div class="hero"><div class="hero-row">
        <div class="hero-main">
          <span class="pill">{pill}</span>
          <h1>{title}</h1><p>{subtitle}</p>
        </div>{meta_html}</div></div>""",
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
            f"""<div class="ec-card"><span class="ico">{icon}</span><h4>{head}</h4><p>{body}</p></div>""",
            unsafe_allow_html=True,
        )


def steps(items):
    for i, (head, body) in enumerate(items, start=1):
        st.markdown(
            f"""<div class="step"><div class="n">{i}</div>
            <div><h5>{head}</h5><p>{body}</p></div></div>""",
            unsafe_allow_html=True,
        )


def status_card_row(items, variant):
    """Like card_row, but with a status-coloured left edge — used on the Coverage
    page (confirmed-detects / confirmed-escapes / unknown). `variant` is one of
    "good", "bad", "warn" (an .ec-card--<variant> modifier)."""
    cols = st.columns(len(items))
    for col, (icon, head, body) in zip(cols, items):
        col.markdown(
            f"""<div class="ec-card ec-card--{variant}">
                <span class="ico">{icon}</span><h4>{head}</h4><p>{body}</p></div>""",
            unsafe_allow_html=True,
        )


def gauge_svg(p_genuine, label, threshold=DEFAULT_THRESHOLD):
    """Semicircular gauge of the raw score, with the decision threshold marked."""
    color = C["genuine"] if label == "Genuine" else C["synthetic"]
    r, cx, cy = 80, 100, 100

    def point(v, radius=r):
        rad = np.deg2rad(180 - 180 * v)
        return cx + radius * np.cos(rad), cy - radius * np.sin(rad)

    x, y = point(p_genuine)
    tx1, ty1 = point(threshold, r - 13)
    tx2, ty2 = point(threshold, r + 13)
    ticks = "".join(
        f'<line x1="{a:.2f}" y1="{b:.2f}" x2="{a2:.2f}" y2="{b2:.2f}" '
        f'stroke="{C["muted"]}" stroke-width="1.5" opacity="0.5"/>'
        for v in (0.0, 0.25, 0.5, 0.75, 1.0)
        for (a, b), (a2, b2) in [(point(v, r - 9), point(v, r + 9))]
    )
    return f"""
    <svg viewBox="0 0 200 138" width="100%" style="max-width:270px">
      <defs>
        <filter id="arcGlow" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="3" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <path d="M 20 100 A {r} {r} 0 0 1 180 100" fill="none"
            stroke="{C['border']}" stroke-width="16" stroke-linecap="round"/>
      {ticks}
      <path d="M 20 100 A {r} {r} 0 0 1 {x:.2f} {y:.2f}" fill="none"
            stroke="{color}" stroke-width="16" stroke-linecap="round"
            filter="url(#arcGlow)" opacity="0.94"/>
      <line x1="{tx1:.2f}" y1="{ty1:.2f}" x2="{tx2:.2f}" y2="{ty2:.2f}"
            stroke="{C['text']}" stroke-width="2.5"/>
      <text x="100" y="58" text-anchor="middle" fill="{C['muted']}"
            font-family="'IBM Plex Mono', monospace" font-size="9.5" letter-spacing="1.6">VERDICT SCORE</text>
      <text x="100" y="90" text-anchor="middle" fill="{color}"
            font-size="30" font-weight="800">{p_genuine*100:.1f}%</text>
      <text x="100" y="109" text-anchor="middle" fill="{C['muted']}"
            font-size="10" letter-spacing="1.1">LIKELIHOOD GENUINE</text>
      <text x="100" y="130" text-anchor="middle" fill="{C['muted']}"
            font-size="10">threshold {threshold:.2f} (marked)</text>
    </svg>
    """


def plot_analysis(y, sr, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), facecolor=C["panel"])
    for ax in (ax1, ax2):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["muted"], labelsize=8)
        for s in ax.spines.values():
            s.set_color(C["border"])

    librosa.display.waveshow(y, sr=sr, ax=ax1, color=C["accent"])
    ax1.fill_between(np.linspace(0, len(y) / sr, num=len(y)), y, color=C["accent"], alpha=0.15, linewidth=0)
    ax1.axhline(0, color=C["border"], linewidth=0.8)
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
    variant = "good" if label == "Genuine" else "bad"
    color = C["genuine"] if label == "Genuine" else C["synthetic"]
    icon = "✅" if label == "Genuine" else "⚠️"
    verdict = "Genuine human voice" if label == "Genuine" else "AI-generated / cloned voice"
    rel = "above" if label == "Genuine" else "at or below"
    thr_note = (f"Score {p_genuine:.3f} is {rel} the {threshold:.2f} threshold."
                + ("" if abs(threshold - DEFAULT_THRESHOLD) < 1e-9
                   else f" (default is {DEFAULT_THRESHOLD:.2f})"))
    # Confidence-scaled tint: a stronger verdict gets a slightly richer wash.
    # Driven by the real margin value, not a decorative constant.
    tint_alpha = 12 + round(margin * 14)  # ~7-15% opacity
    tint = f"{color}{tint_alpha:02x}"
    chips = "".join(
        f'<span class="ec-chip">{html.escape(c)}</span>'
        for c in (f"{duration:.2f}s", f"{sr} Hz", model_name)
    )
    st.markdown(
        f"""
        <div class="ec-card ec-card--{variant} result-reveal"
             style="padding:{SPACE['5']};margin:{SPACE['1']} 0 {SPACE['4']} 0;
             background:linear-gradient(160deg, {tint}, {C['panel']} 70%);">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:{SPACE['4']};flex-wrap:wrap;">
            <div>
              <div style="color:{C['muted']};font-size:{TYPE['xs']};letter-spacing:.6px;text-transform:uppercase;">{html.escape(filename)}</div>
              <div style="color:{color};font-size:{TYPE['h2']};font-weight:800;margin-top:{SPACE['1']};letter-spacing:-.5px;">{icon} {label}</div>
              <div style="color:{C['muted']};font-size:{TYPE['sm']};margin-top:2px;">{verdict}</div>
            </div>
            <div style="text-align:right;">
              <div style="color:{C['text']};font-size:{TYPE['h3']};font-weight:700;">{margin*100:.1f}%</div>
              <div style="color:{C['muted']};font-size:{TYPE['xs']};">confidence margin</div>
            </div>
          </div>
          <div style="background:{C['border']};border-radius:4px;height:6px;margin-top:{SPACE['4']};overflow:hidden;">
            <div style="background:{color};width:{margin*100:.1f}%;height:100%;box-shadow:0 0 10px {color}80;"></div>
          </div>
          <div style="color:{C['muted']};font-size:{TYPE['xs']};margin-top:{SPACE['3']};">{thr_note}</div>
          <div style="margin-top:{SPACE['3']};display:flex;gap:{SPACE['2']};flex-wrap:wrap;">{chips}</div>
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


def analyse(model, model_name, filename, raw_bytes, show_details=True,
            threshold=DEFAULT_THRESHOLD, show_playback=True):
    if show_playback:
        st.audio(raw_bytes)
    y, sr, err = load_audio_safely(raw_bytes)
    if err:
        st.error(f"**{filename}** — {err}")
        return

    with st.spinner("Analysing audio…"):
        label, margin, p_genuine, mfcc = predict(model, y, sr, threshold)

    left, right = st.columns([1, 1.6])
    with left:
        st.markdown(gauge_svg(p_genuine, label, threshold), unsafe_allow_html=True)
    with right:
        result_card(filename, label, margin, len(y) / sr, sr, model_name,
                    threshold, p_genuine)

    if show_details:
        with st.expander("🔬 Show the signal analysis behind this result"):
            with st.container(border=True):
                fig = plot_analysis(y, sr, filename)
                st.pyplot(fig)
                plt.close(fig)
            with st.container(border=True):
                fig2 = plot_mfcc(mfcc)
                st.pyplot(fig2)
                plt.close(fig2)

    st.session_state.history.append({
        "Filename": filename,
        "Prediction": label,
        "Score": f"{p_genuine:.3f}",
        "Threshold": f"{threshold:.2f}",
        "Model": model_name,
        "Time": time.strftime("%H:%M:%S"),
    })


# ============================================================
# PAGES
# ============================================================
def page_overview(models):
    hero("Fraud prevention • Audio forensics",
         "Synthetic Voice Detection",
         "An end-to-end system that listens to a voice clip and determines whether it came from a "
         "real human or an AI voice-cloning engine — built to counter voice-based scams and support "
         "forensic analysis.",
         meta=[("Architecture", "CNN + BiLSTM"), ("Models loaded", str(len(models)))])

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
            <marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3"
                    orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L7,3 z" fill="{C['muted']}"/>
            </marker>
          </defs>
          {''.join(
            f'''<g>
              <rect x="{18 + i*178}" y="30" width="150" height="58" rx="8"
                    fill="{C['panel']}" stroke="{C['border']}" stroke-width="1.2"/>
              <text x="{93 + i*178}" y="55" text-anchor="middle" fill="{C['text']}"
                    font-size="13" font-weight="600">{t}</text>
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
            <rect x="730" y="30" width="150" height="58" rx="8"
                  fill="{C['panel']}" stroke="{C['genuine']}" stroke-width="1.2"/>
            <text x="805" y="55" text-anchor="middle" fill="{C['text']}"
                  font-size="13" font-weight="600">Verdict</text>
            <text x="805" y="72" text-anchor="middle" fill="{C['muted']}"
                  font-size="10.5">genuine / synthetic</text>
          </g>
        </svg>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div class="ec-card ec-card--interactive" style="display:flex;align-items:center;
             justify-content:space-between;gap:{SPACE['4']};flex-wrap:wrap;">
          <div>
            <div style="color:{C['text']};font-weight:600;font-size:{TYPE['body']};">Ready to try it?</div>
            <div style="color:{C['muted']};font-size:{TYPE['sm']};margin-top:{SPACE['1']};">
            Open Detect Voice in the sidebar to upload or record a clip and get a live verdict.</div>
          </div>
          <div style="color:{C['accent']};font-family:'IBM Plex Mono',monospace;font-size:{TYPE['sm']};
               font-weight:600;white-space:nowrap;">Detect Voice →</div>
        </div>""",
        unsafe_allow_html=True,
    )


def page_detect(models):
    hero("Live analysis", "Detect Voice",
         "Upload existing audio files or record directly from your microphone. Every clip stays playable "
         "next to its verdict, so results can be replayed and verified on the spot.",
         meta=[("Models loaded", str(len(models))), ("Session log", str(len(st.session_state.history)))])

    if not models:
        st.error(f"No model file found. Place `{BASELINE_MODEL}` in the app folder and reload.")
        return

    names = list(models.keys())
    default_idx = len(names) - 1

    with st.expander("⚙️  Model & threshold settings", expanded=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            chosen = st.selectbox("Detection model", names, index=default_idx,
                                  help="Compare the original and fine-tuned models on the same clip.")
        with col2:
            details = st.checkbox("Show signal analysis", value=True)

        chosen_path = models[chosen]
        note = MODEL_NOTES.get(chosen_path)
        scores = MODEL_SCORES.get(chosen_path)
        if note and scores:
            asv, itw = scores["asv"], scores["itw"]
            st.markdown(
                f"""<div class="ec-card" style="margin-top:{SPACE['3']};">
                  <div style="color:{C['muted']};font-size:{TYPE['sm']};line-height:1.55;">{note}</div>
                  <div style="display:flex;gap:{SPACE['6']};margin-top:{SPACE['3']};flex-wrap:wrap;">
                    <div><span style="color:{C['muted']};font-size:{TYPE['xs']};">STUDIO AUDIO (ASVspoof)</span><br/>
                      <b style="color:{C['text']};font-size:{TYPE['body']};">{asv[0]*100:.1f}%</b>
                      <span style="color:{C['muted']};font-size:{TYPE['xs']};"> acc &nbsp;·&nbsp;
                      {asv[2]*100:.1f}% prec</span></div>
                    <div><span style="color:{C['muted']};font-size:{TYPE['xs']};">REAL-WORLD (In-the-Wild)</span><br/>
                      <b style="color:{C['text']};font-size:{TYPE['body']};">{itw[0]*100:.1f}%</b>
                      <span style="color:{C['muted']};font-size:{TYPE['xs']};"> acc &nbsp;·&nbsp;
                      {itw[2]*100:.1f}% prec</span></div>
                    <div><span style="color:{C['muted']};font-size:{TYPE['xs']};">EER (studio / real-world)</span><br/>
                      {f'<b style="color:{C["text"]};font-size:{TYPE["body"]};">{scores["eer"][0]*100:.2f}%</b>'
                       f'<span style="color:{C["muted"]};font-size:{TYPE["xs"]};"> / </span>'
                       f'<b style="color:{C["text"]};font-size:{TYPE["body"]};">{scores["eer"][1]*100:.2f}%</b>'
                       if scores.get("eer") else
                       f'<span style="color:{C["muted"]};font-size:{TYPE["sm"]};">not yet measured</span>'}</div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown(f"<div style='height:{SPACE['4']}'></div>", unsafe_allow_html=True)
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

    model = load_model(models[chosen])
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

    # Two-column showcase layout: capture on the left, verdict on the right —
    # nothing here changes the capture/analysis logic, only where it renders.
    capture_col, result_col = st.columns([1, 1.3], gap="large")

    with capture_col:
        mode = option_menu(
            menu_title=None,
            options=["Upload Audio", "Record Live"],
            icons=["cloud-arrow-up-fill", "mic-fill"],
            orientation="horizontal",
            default_index=0,
            key="input_mode",
            styles={
                "container": {
                    "padding": "4px", "background-color": C["panel"],
                    "border": f"1px solid {C['border']}", "border-radius": "8px",
                },
                "icon": {"color": C["accent"], "font-size": "14px"},
                "nav-link": {
                    "font-size": TYPE["sm"], "text-align": "center", "padding": "10px 8px",
                    "border-radius": "6px", "color": C["muted"], "margin": "0 2px",
                    "font-weight": "600",
                },
                "nav-link-selected": {
                    "background-color": C["accent"], "color": "#ffffff", "font-weight": "600",
                },
            },
        )

        # Status strip reflects real session state (nothing simulated): which
        # capture mode is active and where it sits in the record/review flow.
        if mode == "Upload Audio":
            status_color, status_label = C["muted"], "AWAITING INPUT"
        else:
            _pending = st.session_state.pending_recording
            if _pending and not st.session_state.recording_analysed:
                status_color, status_label = C["warn"], "AWAITING CONFIRMATION"
            elif _pending and st.session_state.recording_analysed:
                status_color, status_label = C["genuine"], "ANALYSED"
            else:
                status_color, status_label = C["muted"], "STANDBY — READY TO RECORD"
        st.markdown(
            f"""<div class="status-strip">
              <span class="dot" style="background:{status_color};box-shadow:0 0 8px {status_color}90;"></span>
              <span class="label">{status_label}</span>
            </div>
            <div class="spec-line">TARGET FORMAT · 16 kHz mono · 40×400 MFCC</div>""",
            unsafe_allow_html=True,
        )

        # (filename, raw_bytes, show_playback) tuples ready to be analysed —
        # collected here, rendered in the result column below.
        ready = []

        with st.container(border=True):
            if mode == "Upload Audio":
                st.markdown(
                    f"""<div class="io-caption">
                      <div class="io-glyph">📤</div>
                      <div class="t">Drop a clip to analyse</div>
                      <div class="s">WAV · MP3 · FLAC — multiple files supported</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                files = st.file_uploader("Audio files", type=["wav", "mp3", "flac"],
                                         accept_multiple_files=True, label_visibility="collapsed")
                if files:
                    for f in files:
                        ready.append((f.name, f.getvalue(), True))
                else:
                    st.info("Upload one or more clips to analyse them.")
            else:
                pending = st.session_state.pending_recording
                st.markdown(
                    f"""<div class="io-caption">
                      <div class="io-glyph">🎙️</div>
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
                # result on every rerun, so bytes must be compared, not just truthiness.
                if audio and (pending is None or audio["bytes"] != pending["bytes"]):
                    st.session_state.pending_recording = {
                        "bytes": audio["bytes"],
                        "name": f"live_recording_{time.strftime('%H%M%S')}.wav",
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
                        ready.append((pending["name"], pending["bytes"], False))

    with result_col:
        if ready:
            for filename, raw_bytes, show_playback in ready:
                st.markdown(f"#### {filename}")
                analyse(model, chosen, filename, raw_bytes, details, threshold,
                        show_playback=show_playback)
                st.divider()
        else:
            st.markdown(
                f"""<div class="ec-card" style="text-align:center;padding:{SPACE['7']} {SPACE['5']};">
                  <div style="font-size:26px;margin-bottom:{SPACE['3']};">🗂️</div>
                  <div style="color:{C['text']};font-weight:600;font-size:{TYPE['body']};">No result yet</div>
                  <div style="color:{C['muted']};font-size:{TYPE['sm']};margin-top:{SPACE['1']};">
                  Upload or record a clip on the left to see its verdict here.</div>
                </div>""",
                unsafe_allow_html=True,
            )


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
        ("Conv2D", "32 filters", 62, C['muted']),
        ("MaxPool", "÷2", 50, C['muted']),
        ("BiLSTM", "64×2 hidden", 74, C['muted']),
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
    st.markdown(
        f"""<div style="display:flex;gap:{SPACE['5']};flex-wrap:wrap;margin:{SPACE['2']} 0 {SPACE['1']} 0;">
          <span style="color:{C['muted']};font-size:{TYPE['xs']};">
            <span style="color:{C['accent']};">■</span> feature extraction</span>
          <span style="color:{C['muted']};font-size:{TYPE['xs']};">
            <span style="color:{C['muted']};">■</span> deeper representation</span>
          <span style="color:{C['muted']};font-size:{TYPE['xs']};">
            <span style="color:{C['genuine']};">■</span> decision</span>
        </div>""",
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

    st.markdown("### Final model — measured on both domains")
    fin = MODEL_SCORES[COMBINED_V2_MODEL]
    stat_row([(f"{fin['asv'][0]*100:.1f}%", "Studio accuracy"),
              (f"{fin['itw'][0]*100:.1f}%", "Real-world accuracy"),
              (f"{fin['asv'][2]*100:.1f}%", "Studio precision"),
              (f"{fin['itw'][2]*100:.1f}%", "Real-world precision")])
    st.caption("Combined + classical-TTS model, evaluated on ASVspoof 2019 LA dev (24,844 studio "
               "clips) and In-the-Wild validation (6,355 real-world clips). EER not yet computed "
               "for this model — the figures below are for the three earlier versions.")

    st.markdown("### Four-stage comparison")
    st.caption("Every model evaluated on both held-out test sets. Bona-fide = genuine human voice.")

    def _fmt(triple):
        return f"{triple[0]*100:.1f}% / {triple[1]*100:.1f}% / {triple[2]*100:.1f}%"

    order = [BASELINE_MODEL, FINETUNED_MODEL, COMBINED_MODEL, COMBINED_V2_MODEL]
    st.table({
        "Model": ["1 · Baseline (ASVspoof only)",
                  "2 · Fine-tuned (In-the-Wild only)",
                  "3 · Combined (both datasets)",
                  "4 · Combined + classical TTS"],
        "Studio — acc / recall / precision": [_fmt(MODEL_SCORES[m]["asv"]) for m in order],
        "Real-world — acc / recall / precision": [_fmt(MODEL_SCORES[m]["itw"]) for m in order],
    })

    st.markdown("### Equal Error Rate")
    st.caption("EER is threshold-independent: the rate at which false acceptances and false "
               "rejections are equal. Lower is better; 50% would be random guessing. Not yet "
               "computed for model 4 — added here once that pass is run.")
    eer_order = [m for m in order if MODEL_SCORES[m].get("eer")]
    st.table({
        "Model": ["1 · Baseline (ASVspoof only)",
                  "2 · Fine-tuned (In-the-Wild only)",
                  "3 · Combined (both datasets)"][:len(eer_order)],
        "ASVspoof EER": [f"{MODEL_SCORES[m]['eer'][0]*100:.2f}%" for m in eer_order],
        "In-the-Wild EER": [f"{MODEL_SCORES[m]['eer'][1]*100:.2f}%" for m in eer_order],
    })
    st.markdown(
        f"""<div class="banner"><b>Reading the EER table.</b> The baseline's real-world EER of 41.52%
        is close to the 50% of random guessing — on everyday recordings it could not separate genuine
        from cloned speech at <i>any</i> threshold, so retraining was necessary rather than optional.
        Joint training cost 2.64 points of ASVspoof EER (5.83% → 8.47%) and returned 37.96 points of
        real-world EER (41.52% → 3.56%), roughly a twelvefold improvement in deployment conditions.</div>""",
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


def page_report():
    hero("Evidence trail", "Session Report",
         "Every clip analysed in this session, in order, with the model used — exportable as a CSV "
         "for inclusion in project documentation.")

    if not st.session_state.history:
        st.markdown(
            f"""<div class="ec-card" style="text-align:center;padding:{SPACE['7']} {SPACE['5']};">
              <div style="font-size:26px;margin-bottom:{SPACE['3']};">📭</div>
              <div style="color:{C['text']};font-weight:600;font-size:{TYPE['body']};">No clips analysed yet</div>
              <div style="color:{C['muted']};font-size:{TYPE['sm']};margin-top:{SPACE['1']};">
              Head to Detect Voice to run your first analysis — results will appear here, ready to export.</div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    rows = st.session_state.history
    genuine = sum(1 for r in rows if r["Prediction"] == "Genuine")
    stat_row([(str(len(rows)), "Clips analysed"),
              (str(genuine), "Genuine"),
              (str(len(rows) - genuine), "Synthetic")])

    st.markdown("### Detailed log")
    cols = ["Filename", "Prediction", "Score", "Threshold", "Model", "Time"]

    def _badge(label):
        color = C["genuine"] if label == "Genuine" else C["synthetic"]
        return (f'<span style="display:inline-block;padding:2px 9px;border-radius:4px;'
                f'background:{color}1f;color:{color};font-weight:600;font-size:{TYPE["xs"]};">'
                f'{html.escape(label)}</span>')

    head_cells = "".join(f'<th style="padding:{SPACE["3"]} {SPACE["4"]};">{c}</th>' for c in cols)
    body_rows = "".join(
        "<tr>" + "".join(
            f'<td style="padding:{SPACE["3"]} {SPACE["4"]};border-top:1px solid {C["border"]};'
            f'font-size:{TYPE["sm"]};color:{C["text"]};">'
            + (_badge(r.get(c, "")) if c == "Prediction" else html.escape(str(r.get(c, ""))))
            + "</td>"
            for c in cols
        ) + "</tr>"
        for r in rows
    )
    st.markdown(
        f"""<div class="ec-card" style="padding:0;overflow-x:auto;">
          <table style="width:100%;border-collapse:collapse;white-space:nowrap;">
            <thead><tr>{head_cells}</tr></thead>
            <tbody>{body_rows}</tbody>
          </table>
        </div>""",
        unsafe_allow_html=True,
    )

    csv = ",".join(cols) + "\n" + "\n".join(
        ",".join(str(r.get(c, "")).replace(",", ";") for c in cols) for r in rows
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        st.download_button("⬇️  Export CSV", csv, file_name="detection_report.csv",
                           use_container_width=True)
    with c2:
        if st.button("🗑️  Clear session", use_container_width=True):
            st.session_state.history = []
            st.rerun()


def page_coverage():
    hero("Scope & honesty", "Detection Coverage",
         "Every real detector has a boundary. This page states it explicitly: what's "
         "measured and working, what's measured and failing, and what's genuinely "
         "still unknown — rather than a single blanket claim of \"detects synthetic voices.\"")

    stat_row([
        ("4", "Confirmed working"),
        ("2", "Confirmed gaps"),
        ("2", "Untested / unknown"),
    ])

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
    ], "good")

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
    ], "bad")

    st.markdown("### ❓ Unknown — never actually tested")
    status_card_row([
        ("✨", "Modern commercial cloning tools (2023+)",
         "Both training datasets predate today's most advanced diffusion/flow-based "
         "cloning engines. No reference for their artifacts exists in training."),
        ("📞", "Telephone-codec compressed audio",
         "8kHz G.711/GSM — the actual medium of real scam calls. Never trained or "
         "tested on codec-degraded audio at all."),
    ], "warn")

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


# ============================================================
# APP
# ============================================================
st.set_page_config(page_title="Synthetic Voice Detection", page_icon="🎙️", layout="wide")
inject_css()

if "history" not in st.session_state:
    st.session_state.history = []
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

models = available_models()

with st.sidebar:
    st.markdown(
        f"""<div style="padding:{SPACE['1']} 2px {SPACE['5']} 2px;">
          <div style="font-size:18px;font-weight:700;letter-spacing:-.2px;color:{C['text']};">VoiceGuard</div>
          <div style="color:{C['muted']};font-family:'IBM Plex Mono',monospace;
               font-size:{TYPE['xs']};letter-spacing:1px;text-transform:uppercase;margin-top:{SPACE['1']};">
          Synthetic Voice Detection</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # "---" entries render as a native <hr> divider (non-selectable, never
    # returned as the chosen value) — grouping the 7 pages into three clusters:
    # Analyze / Understand / Report.
    selected = option_menu(
        menu_title=None,
        options=["Overview", "Detect Voice", "---",
                 "How It Works", "Model & Results", "Coverage", "---",
                 "Session Report", "About"],
        icons=["grid-1x2", "soundwave", "",
               "diagram-3", "graph-up", "shield-check", "",
               "clipboard-data", "info-circle"],
        default_index=0,
        styles={
            # This component renders inside its OWN embedded iframe, with its own
            # default (white) page background — "transparent" here reveals THAT,
            # not our dark sidebar behind it. Must be an explicit dark fill.
            "container": {"padding": "0", "background-color": C["panel"]},
            "icon": {"color": C["accent"], "font-size": "14px"},
            "nav-link": {
                "font-size": "13px", "text-align": "left", "margin": "2px 0",
                "padding": "10px 12px", "border-radius": "6px", "color": C["muted"],
                "background-color": C["panel"], "--hover-color": C["panel2"],
            },
            "nav-link-selected": {
                "background-color": C["panel2"], "color": C["text"],
                "font-weight": "600", "border-left": f"2px solid {C['accent']}",
                "border-radius": "6px",
            },
            "separator": {"background-color": C["border"], "margin": "8px 0"},
        },
    )

    st.markdown("---")
    st.markdown(
        f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:{TYPE['xs']};
             color:{C['muted']};letter-spacing:1px;text-transform:uppercase;
             margin-bottom:{SPACE['3']};">System status</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:{TYPE['sm']};
             color:{C['muted']};line-height:2.1;">
          <div style="display:flex;justify-content:space-between;">
            <span>COMPUTE</span><b style="color:{C['text']}">{device.upper()}</b></div>
          <div style="display:flex;justify-content:space-between;">
            <span>MODELS</span><b style="color:{C['text']}">{len(models)}</b></div>
          <div style="display:flex;justify-content:space-between;">
            <span>ANALYSED</span><b style="color:{C['text']}">{len(st.session_state.history)}</b></div>
        </div>""",
        unsafe_allow_html=True,
    )

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
