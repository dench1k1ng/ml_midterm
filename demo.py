import json
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parent / "artifacts" / "participant3" / "demo_model"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, use_safetensors=True,
    ).eval()
    config = json.loads((MODEL_DIR / "inference.json").read_text())
    return tokenizer, model, config


st.set_page_config(page_title="SMS Spam Detector", page_icon="✉️")
st.title("SMS Spam Detector")
with st.form("sms"):
    message = st.text_area("SMS message", placeholder="Paste an English SMS", max_chars=5000)
    submitted = st.form_submit_button("Classify", type="primary")

if submitted:
    if not message.strip():
        st.warning("Enter an SMS message.")
    elif not (MODEL_DIR / "inference.json").is_file():
        st.error("Run notebook sections 2 and 3.4 to export the trained model, then try again.")
    else:
        tokenizer, model, config = load_model()
        inputs = tokenizer(
            message.strip(), padding="max_length", truncation=True,
            max_length=config["max_length"], return_tensors="pt",
        )
        with torch.inference_mode():
            score = model(**inputs).logits.softmax(dim=1)[0, 1].item()
        threshold = config["threshold"]
        if score >= threshold:
            st.warning("Spam")
        else:
            st.success("Ham")
        st.metric("Spam probability", f"{score:.1%}")
        st.caption(f"BERT-Tiny · Decision threshold {threshold:.1%} · Uncalibrated probability")
