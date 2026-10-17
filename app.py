"""Art-direction-inspired Streamlit UI for The Model That Doesn't Know.

The visual language is inspired by ArtCraft's editorial typography, numbered
sections, confident high-contrast layout, and moving feature ticker. This is an
original interface; model inference and threshold semantics remain unchanged.
"""

from pathlib import Path
import json
import html

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CHECKPOINT = Path("outputs/best_model.pt")
RESULTS = Path("outputs/evaluation_results.json")
KNOWN_CLASSES = ["airplane", "automobile", "bird", "cat", "deer"]

st.set_page_config(
    page_title="The Model That Doesn't Know",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------- Art direction -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
      --paper: #f3f0e9;
      --ink: #171815;
      --muted: #66685f;
      --line: #c9c7bd;
      --acid: #d7fb53;
      --blue: #6c72e8;
      --coral: #f0785d;
      --panel: #e8e5dc;
    }

    .stApp {
      background: var(--paper);
      background-image: radial-gradient(circle at 88% 8%, rgba(108,114,232,.10), transparent 25%), radial-gradient(circle at 4% 45%, rgba(240,120,93,.08), transparent 22%);
      color: var(--ink);
      font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .block-container {
      max-width: 1320px;
      padding-top: 1.2rem;
      padding-bottom: 3.5rem;
    }
    h1, h2, h3, h4, p, label { color: var(--ink); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
    .mono {
      font-family: 'DM Mono', monospace;
      text-transform: uppercase;
      letter-spacing: .08em;
      font-size: .72rem;
    }
    .topbar {
      display:flex; align-items:center; justify-content:space-between;
      border-bottom:1px solid var(--ink); padding: .3rem 0 1rem;
      margin-bottom: 2.5rem;
    }
    .brand { font:700 1.08rem 'Space Grotesk',sans-serif; letter-spacing:-.05em; }
    .brand-dot { display:inline-block; width:10px; height:10px; border-radius:50%;
      background:var(--coral); margin-right:9px; animation:dotPulse 1.8s ease-in-out infinite; box-shadow:0 0 0 0 rgba(240,120,93,.55); }
    .top-tag { font: .7rem 'DM Mono',monospace; letter-spacing:.08em; }
    .hero {
      display:grid; grid-template-columns: minmax(0,1.45fr) minmax(230px,.55fr);
      gap:2rem; align-items:end; padding:0 0 2.2rem;
      border-bottom:1px solid var(--ink); position:relative;
    }
    .kicker { font: .72rem 'DM Mono',monospace; text-transform:uppercase;
      letter-spacing:.12em; margin-bottom:1.1rem; }
    .hero h1 {
      font-size:clamp(3.3rem,7vw,7rem); line-height:.89; letter-spacing:-.085em;
      margin:0; font-weight:600; max-width:850px;
    }
    .hero h1 em { font-style:normal; color:var(--blue); display:inline-block; }
    .hero::after { content:''; position:absolute; right:0; top:0; width:clamp(70px,12vw,160px); height:clamp(70px,12vw,160px); border:1px solid rgba(108,114,232,.25); border-radius:50%; pointer-events:none; animation:orbit 18s linear infinite; }
    .hero::before { content:''; position:absolute; right:clamp(30px,5vw,65px); top:clamp(25px,4vw,55px); width:8px; height:8px; border-radius:50%; background:var(--coral); box-shadow:0 0 0 8px rgba(240,120,93,.12), 0 0 0 16px rgba(240,120,93,.06); animation:float 3s ease-in-out infinite; }
    .hero-side { border-left:1px solid var(--ink); padding-left:1.25rem; }
    .hero-side p { font-size:1rem; line-height:1.6; margin:.5rem 0 1rem; max-width:330px; }
    .pill {
      display:inline-block; border:1px solid var(--ink); border-radius:999px;
      padding:.42rem .7rem; font: .67rem 'DM Mono',monospace;
      text-transform:uppercase; letter-spacing:.04em;
    }
    .ticker {
      overflow:hidden; white-space:nowrap; border-bottom:1px solid var(--ink);
      padding:.8rem 0; margin-bottom:2.3rem;
    }
    .ticker-track { display:inline-block; min-width:200%; animation:marquee 18s linear infinite;
      font: .74rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.08em; }
    .ticker-track span { padding:0 1.4rem; }
    .ticker:hover .ticker-track { animation-play-state:paused; }
    .ticker-track b { color:var(--blue); font-weight:500; }
    .section-head {
      display:flex; align-items:baseline; justify-content:space-between;
      border-bottom:1px solid var(--ink); padding-bottom:.8rem; margin:2.2rem 0 1.2rem;
    }
    .section-head h2 { font-size:1.65rem; letter-spacing:-.055em; margin:0; }
    .section-head .mono { color:var(--muted); }
    .panel {
      background:var(--panel); border:1px solid var(--ink); border-radius:3px;
      padding:1.15rem; height:100%; transition:transform .25s ease, box-shadow .25s ease;
    }
    .panel:hover { transform:translateY(-3px); box-shadow:5px 5px 0 rgba(23,24,21,.12);
    }
    .panel-dark { background:var(--ink); color:var(--paper); border:1px solid var(--ink);
      padding:1.4rem; min-height:100%; }
    .panel-dark h3, .panel-dark p, .panel-dark div { color:var(--paper); }
    .panel-label { font:.68rem 'DM Mono',monospace; text-transform:uppercase;
      letter-spacing:.1em; margin-bottom:1rem; }
    .upload-title { font:600 1.55rem 'Space Grotesk',sans-serif; letter-spacing:-.05em;
      line-height:1.05; margin:.35rem 0 .75rem; }
    .upload-note { color:var(--muted); font-size:.88rem; line-height:1.55; }
    div[data-testid="stFileUploader"] {
      background:#f8f6f0; border:1px dashed #6c6d63; border-radius:3px; padding:.65rem; transition:border-color .2s ease, box-shadow .2s ease, transform .2s ease;
    }
    div[data-testid="stFileUploader"]:hover { border-color:var(--blue); box-shadow:0 0 0 4px rgba(108,114,232,.10); transform:translateY(-2px); }
    div[data-testid="stFileUploader"] section { background:transparent; }
    div[data-testid="stFileUploader"] label { color:var(--ink)!important; }
    div[data-testid="stSelectbox"] label { color:var(--ink)!important;
      font: .68rem 'DM Mono',monospace!important; text-transform:uppercase; letter-spacing:.08em; }
    div[data-testid="stSelectbox"] > div > div {
      background:#f8f6f0; border:1px solid var(--ink); border-radius:3px; color:var(--ink);
    }
    div[data-testid="stButton"] button {
      background:var(--acid); color:var(--ink); border:1px solid var(--ink);
      border-radius:3px; font:600 .88rem 'Space Grotesk',sans-serif;
      min-height:3rem; width:100%; transition:transform .18s ease, box-shadow .18s ease;
    }
    div[data-testid="stButton"] button:hover {
      color:var(--ink); border:1px solid var(--ink); transform:translateY(-2px);
      box-shadow:4px 4px 0 var(--ink);
    }
    .result-banner {
      padding:1.2rem; border:1px solid var(--ink); margin-bottom:1rem;
      transform-origin:top center;
    }
    .result-banner.known { background:var(--acid); }
    .result-banner.unknown { background:var(--coral); }
    .result-status { font:.7rem 'DM Mono',monospace; letter-spacing:.12em; text-transform:uppercase; }
    .result-class { font:600 clamp(2.3rem,4vw,4rem) 'Space Grotesk',sans-serif;
      letter-spacing:-.08em; line-height:1; margin:.65rem 0; overflow-wrap:anywhere; }
    .result-caption { font-size:.86rem; line-height:1.5; max-width:550px; }
    .score-grid { display:grid; grid-template-columns:1fr 1fr; gap:.7rem; margin:.8rem 0 1.2rem; }
    .score-box { border:1px solid #45463f; padding:.8rem; }
    .score-label { color:#c7c7bc!important; font:.64rem 'DM Mono',monospace;
      text-transform:uppercase; letter-spacing:.07em; }
    .score-value { color:#fff!important; font:600 1.55rem 'Space Grotesk',sans-serif; margin-top:.35rem; }
    .prob-row { margin:.9rem 0; }
    .prob-top { display:flex; justify-content:space-between; gap:1rem;
      font-size:.82rem; margin-bottom:.35rem; }
    .prob-track { height:7px; background:#41423c; width:100%; }
    .prob-fill { height:7px; background:var(--acid); transform-origin:left; box-shadow:0 0 12px rgba(215,251,83,.18); }
    /* Scroll-triggered reveals: animation progresses as each section enters the viewport. */
    .scroll-reveal { opacity:1; transform:none; }
    @supports (animation-timeline: view()) {
      .scroll-reveal { animation: scrollReveal linear both; animation-timeline: view(); animation-range: entry 0% entry 55%; }
      .prob-fill { animation: barReveal linear both; animation-timeline: view(); animation-range: entry 0% entry 65%; }
    }
    .small-note { color:var(--muted); font-size:.78rem; line-height:1.55; }
    .feature-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem; }
    .feature-card { border:1px solid var(--ink); padding:1rem; min-height:150px;
      transition:background .2s ease, transform .2s ease; }
    .feature-card:hover { background:var(--acid); transform:translateY(-5px) rotate(-.3deg); box-shadow:5px 5px 0 var(--ink); }
    .feature-num { font:.68rem 'DM Mono',monospace; color:var(--blue); }
    .feature-card h3 { font-size:1.15rem; margin:.8rem 0 .4rem; letter-spacing:-.04em; }
    .feature-card p { font-size:.8rem; line-height:1.5; color:var(--muted); margin:0; }
    .footer {
      border-top:1px solid var(--ink); margin-top:3rem; padding-top:1rem;
      display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap;
      font:.68rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.05em;
    }
    div[data-testid="stAlert"] { border-radius:3px; }
    @keyframes scrollReveal { from {opacity:.12; transform:translateY(42px); filter:blur(3px)} to {opacity:1; transform:translateY(0); filter:blur(0)} }
    @keyframes barReveal { from {clip-path:inset(0 100% 0 0)} to {clip-path:inset(0 0 0 0)} }
    @keyframes dotPulse { 0%,100% {box-shadow:0 0 0 0 rgba(240,120,93,.5)} 50% {box-shadow:0 0 0 8px rgba(240,120,93,0)} }
    .hero:hover::after { transform:scale(1.06); transition:transform .6s ease; }
    .hero:hover::before { transform:scale(1.25); transition:transform .3s ease; }
    @keyframes marquee { from {transform:translateX(0)} to {transform:translateX(-50%)} }
    @keyframes grow { from {width:0} }
    @media (max-width: 800px) {
      .hero { grid-template-columns:1fr; gap:1.2rem; }
      .hero-side { border-left:0; border-top:1px solid var(--ink); padding:1rem 0 0; }
      .hero h1 { font-size:clamp(3rem,15vw,5rem); }
      .feature-grid { grid-template-columns:1fr; }
      .top-tag { display:none; }
      .block-container { padding-top:.8rem; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation-duration:.01ms!important; transition-duration:.01ms!important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Model -----------------------------------
@st.cache_resource
def load_artifacts():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT}. Run train.py first or place best_model.pt in outputs/."
        )
    if not RESULTS.exists():
        raise FileNotFoundError(
            f"Evaluation results not found at {RESULTS}. Run evaluate.py after training."
        )

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    with RESULTS.open("r", encoding="utf-8") as f:
        results = json.load(f)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN_CLASSES))
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return model, device, tf, results


# ------------------------------- Header ----------------------------------
st.markdown(
    """
    <section class="hero scroll-reveal">
      <div>
        <div class="kicker">01 / The model that doesn't know</div>
        <h1>Not every answer<br>deserves <em>certainty.</em></h1>
      </div>
      <div class="hero-side">
        <div class="mono">A small experiment in machine uncertainty</div>
        <p>Five familiar classes. Five classes held back. One model asked to know the difference — and admit when it might not.</p>
        <span class="pill">MSP + Energy scoring</span>
      </div>
    </section>
    <div class="ticker"><div class="ticker-track">
      <span>Recognise what you know</span><b>✳</b><span>Measure uncertainty</span><b>✳</b>
      <span>Reject when the signal is weak</span><b>✳</b><span>Never confuse confidence with truth</span><b>✳</b>
      <span>Recognise what you know</span><b>✳</b><span>Measure uncertainty</span><b>✳</b>
      <span>Reject when the signal is weak</span><b>✳</b><span>Never confuse confidence with truth</span><b>✳</b>
    </div></div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-head scroll-reveal">
      <h2>Run an image through the model</h2>
      <span class="mono">02 / Live inference</span>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, device, transform, results = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.markdown(
        """
        <div class="panel scroll-reveal">
          <div class="panel-label">Input / 01</div>
          <div class="upload-title">Bring an image.<br>Test the boundary.</div>
          <p class="upload-note">Upload a photo and compare the model's strongest known-class prediction against a validation-derived rejection threshold.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    method = st.selectbox("Detection method", ["MSP", "Energy"], help="Thresholds are loaded from evaluation_results.json.")
    uploaded = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        help="Use a clear image for a better visual test. CIFAR-10 itself contains 32×32 images.",
    )
    if uploaded is None:
        st.markdown(
            """
            <div class="panel" style="margin-top:12px; min-height:170px; display:flex; flex-direction:column; justify-content:center;">
              <div class="mono">Waiting for input</div>
              <div style="font:600 1.6rem 'Space Grotesk',sans-serif; letter-spacing:-.05em; margin:.6rem 0;">The next image is a question.</div>
              <div class="small-note">Try an airplane, automobile, bird, cat or deer — then try something outside those classes.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    if uploaded is None:
        st.markdown(
            """
            <div class="panel-dark scroll-reveal" style="min-height:390px; display:flex; flex-direction:column; justify-content:space-between;">
              <div>
                <div class="panel-label">System / 02</div>
                <div style="font:600 clamp(2.3rem,4vw,4.2rem) 'Space Grotesk',sans-serif; letter-spacing:-.08em; line-height:.98;">
                  A prediction<br>is not proof.
                </div>
              </div>
              <div>
                <div class="score-grid">
                  <div class="score-box"><div class="score-label">Known labels</div><div class="score-value">05</div></div>
                  <div class="score-box"><div class="score-label">Held-out labels</div><div class="score-value">05</div></div>
                </div>
                <div class="small-note" style="color:#c7c7bc!important;">The classifier always scores the five known labels. The rejection rule decides whether to trust that forced choice.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            image = Image.open(uploaded).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x).cpu()
                probs = torch.softmax(logits, dim=1)[0].numpy()

            msp = float(probs.max())
            thresholds = results["threshold_calibration"]
            temperature = float(thresholds.get("temperature", 1.0))
            energy_id_score = float(
                (temperature * torch.logsumexp(logits / temperature, dim=1)).item()
            )

            if method == "MSP":
                score = msp
                threshold = float(thresholds["msp_threshold"])
                score_name = "Maximum softmax probability"
            else:
                score = energy_id_score
                threshold = float(thresholds["energy_id_score_threshold"])
                score_name = "Energy ID score"

            predicted_idx = int(np.argmax(probs))
            predicted_class = KNOWN_CLASSES[predicted_idx]
            rejected = score < threshold
            banner_class = "unknown" if rejected else "known"
            status = "Potentially unknown" if rejected else "Accepted as known"
            explanation = (
                "The selected score fell below the validation-derived threshold. The class label is shown for reference, but should not be trusted."
                if rejected else
                "The selected score met the validation-derived threshold. This is not proof that the image belongs to a known class."
            )

            st.image(image, caption=f"Input image · {image.width} × {image.height}", use_container_width=True)
            st.markdown(
                f"""
                <div class="result-banner {banner_class}">
                  <div class="result-status">{html.escape(status)}</div>
                  <div class="result-class">{html.escape(predicted_class.title())}</div>
                  <div class="result-caption">{html.escape(explanation)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="panel-dark scroll-reveal">
                  <div class="panel-label">Decision metrics / 03</div>
                  <div class="score-grid">
                    <div class="score-box">
                      <div class="score-label">Selected score</div>
                      <div class="score-value">{score:.4f}</div>
                    </div>
                    <div class="score-box">
                      <div class="score-label">Rejection threshold</div>
                      <div class="score-value">{threshold:.4f}</div>
                    </div>
                  </div>
                  <div class="small-note" style="color:#c7c7bc!important;">Method: {html.escape(score_name)} · Device: {html.escape(str(device).upper())}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="section-head scroll-reveal"><h2>Class scores</h2><span class="mono">03 / Relative probabilities</span></div>',
                unsafe_allow_html=True,
            )
            sorted_idx = np.argsort(probs)[::-1]
            for rank, idx in enumerate(sorted_idx[:3], start=1):
                name = KNOWN_CLASSES[int(idx)]
                probability = float(probs[int(idx)])
                st.markdown(
                    f"""
                    <div class="prob-row">
                      <div class="prob-top"><span>{rank:02d} &nbsp; {html.escape(name.title())}</span><span>{probability*100:.1f}%</span></div>
                      <div class="prob-track"><div class="prob-fill" style="width:{probability*100:.2f}%; animation-delay:{(rank-1)*0.14:.2f}s"></div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with st.expander("Show all five class probabilities"):
                for idx in np.argsort(probs)[::-1]:
                    st.write(f"**{KNOWN_CLASSES[int(idx)].title()}** — {float(probs[int(idx)]):.4f}")
            st.caption(
                "Softmax values are relative probabilities over the five known classes, not calibrated probabilities of being correct. OOD detection can fail."
            )
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")

st.markdown(
    """
    <div class="section-head scroll-reveal">
      <h2>What this experiment tests</h2>
      <span class="mono">04 / Method notes</span>
    </div>
    <div class="feature-grid scroll-reveal">
      <div class="feature-card">
        <div class="feature-num">01 / KNOWN CLASSES</div>
        <h3>Recognise the familiar</h3>
        <p>Airplane, automobile, bird, cat and deer are the only output labels the classifier can predict.</p>
      </div>
      <div class="feature-card">
        <div class="feature-num">02 / REJECTION</div>
        <h3>Measure the score</h3>
        <p>MSP or the energy ID score is compared with a threshold calibrated on known validation examples.</p>
      </div>
      <div class="feature-card">
        <div class="feature-num">03 / LIMITATIONS</div>
        <h3>Confidence can fail</h3>
        <p>Unfamiliar images may still receive high scores, while valid known images can be rejected.</p>
      </div>
    </div>
    <div class="footer">
      <span>THE MODEL THAT DOESN'T KNOW</span>
      <span>ResNet-18 · CIFAR-10 · Open-set classification</span>
      <span>Confidence ≠ certainty</span>
    </div>
    """,
    unsafe_allow_html=True,
)
