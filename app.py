"""Streamlit demo for open-set classification."""
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CHECKPOINT = Path("outputs/best_model.pt")
RESULTS = Path("outputs/evaluation_results.json")
KNOWN_CLASSES = ["airplane", "automobile", "bird", "cat", "deer"]


@st.cache_resource
def load_artifacts():
    if not CHECKPOINT.exists():
        raise FileNotFoundError("Checkpoint not found. Run train.py first.")
    if not RESULTS.exists():
        raise FileNotFoundError("Thresholds not found. Run evaluate.py after training.")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    import json
    with open(RESULTS) as f:
        results = json.load(f)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN_CLASSES))
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return model, device, tf, results


st.set_page_config(page_title="The Model That Doesn't Know", page_icon="🔎", layout="centered")
st.title("The Model That Doesn't Know")
st.write("Upload an image. The classifier predicts one of five known CIFAR-10 categories, "
         "or flags the input as potentially unfamiliar when its score is below a validation-derived threshold.")
st.caption("Known classes: airplane, automobile, bird, cat, deer. The detector is not a guarantee that every unknown will be caught.")

try:
    model, device, transform, results = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

method = st.selectbox("Uncertainty method", ["MSP", "Energy"])
uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp", "bmp"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).cpu()
        probs = torch.softmax(logits, dim=1)[0].numpy()
        msp = float(probs.max())
        temperature = float(results["threshold_calibration"].get("temperature", 1.0))
        energy_id_score = float((temperature * torch.logsumexp(logits / temperature, dim=1)).item())
    thresholds = results["threshold_calibration"]
    if method == "MSP":
        score = msp
        threshold = thresholds["msp_threshold"]
        score_name = "Maximum softmax probability"
    else:
        score = energy_id_score
        threshold = thresholds["energy_id_score_threshold"]
        score_name = "Energy ID score"
    predicted_idx = int(np.argmax(probs))
    st.subheader("Prediction")
    st.metric("Top predicted class", KNOWN_CLASSES[predicted_idx])
    st.write(f"{score_name}: **{score:.4f}**")
    st.write(f"Rejection threshold: **{threshold:.4f}**")
    if score < threshold:
        st.warning("Potentially unknown / uncertain input — do not trust the forced class prediction.")
    else:
        st.success(f"Accepted as a known class: **{KNOWN_CLASSES[predicted_idx]}**")
    st.subheader("Known-class probabilities")
    st.bar_chart({name: float(prob) for name, prob in zip(KNOWN_CLASSES, probs)})
    st.info("Softmax probabilities are relative to the five known classes. A high score does not prove "
            "that an image is in-distribution; this demo uses a threshold calibrated on known validation images.")
