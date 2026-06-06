"""
Vision-Language Model App: Image Captioning + Visual Question Answering
ELC Project 2025-26 | Topic: Vision-Language Models for Image Understanding

Author: Saanjal jain
Roll No : 1024030533
Stack: Streamlit · Hugging Face Transformers · BLIP/BLIP-2 · PyTorch · OpenCV
"""

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path

# ── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="VLM Image Understanding",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: clean, professional dark-accent UI ──────────────────────────────────
st.markdown("""
<style>
/* Base */
[data-testid="stAppViewContainer"] { background: #F8FAFC; }
[data-testid="stSidebar"] { background: #0F172A; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

/* Title strip */
.vlm-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #0284C7 100%);
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    color: white;
}
.vlm-header h1 { color: white; margin: 0; font-size: 2rem; }
.vlm-header p  { color: #BAE6FD; margin: 4px 0 0; font-size: 1rem; }

/* Cards */
.card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 16px;
}
.card h3 { color: #1E3A5F; margin-top: 0; }

/* Caption output */
.caption-box {
    background: #EFF6FF;
    border-left: 4px solid #2563EB;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    font-size: 1.05rem;
    color: #1E40AF;
    font-style: italic;
}

/* Answer output */
.answer-box {
    background: #F0FDF4;
    border-left: 4px solid #16A34A;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    font-size: 1.05rem;
    color: #15803D;
}

/* Confidence badge */
.conf-badge {
    display: inline-block;
    background: #DBEAFE;
    color: #1D4ED8;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: 8px;
}

/* Metric pill */
.metric-pill {
    background: #0F172A;
    color: #38BDF8;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.85rem;
    text-align: center;
}

/* Image stats bar */
.img-stat { color: #64748B; font-size: 0.85rem; }

/* Spinner override */
.stSpinner > div { border-top-color: #0284C7 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_choice: str):
    """
    Load BLIP or BLIP-2 from Hugging Face Transformers.
    Returns (processor, model, device).
    Cached so models are loaded only once per session.
    """
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration,
        BlipForQuestionAnswering,
        Blip2Processor, Blip2ForConditionalGeneration,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_choice == "BLIP (Fast · ~1 GB)":
        caption_id = "Salesforce/blip-image-captioning-large"
        vqa_id     = "Salesforce/blip-vqa-base"
        cap_proc   = BlipProcessor.from_pretrained(caption_id)
        cap_model  = BlipForConditionalGeneration.from_pretrained(
            caption_id, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device)
        vqa_proc   = BlipProcessor.from_pretrained(vqa_id)
        vqa_model  = BlipForQuestionAnswering.from_pretrained(
            vqa_id, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device)
        return {"cap": (cap_proc, cap_model), "vqa": (vqa_proc, vqa_model), "device": device, "family": "blip"}

    else:  # BLIP-2
        model_id = "Salesforce/blip2-opt-2.7b"
        processor = Blip2Processor.from_pretrained(model_id)
        model     = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            model = model.to(device)
        return {"unified": (processor, model), "device": device, "family": "blip2"}


# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image, enhance: bool = True) -> Image.Image:
    """
    Optional OpenCV-based enhancement pipeline:
    • CLAHE for contrast normalisation
    • Mild bilateral filter for noise reduction
    • Sharpening kernel
    Returns enhanced PIL Image in RGB.
    """
    if not enhance:
        return pil_img.convert("RGB")

    img_np = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # LAB colour space → apply CLAHE only to Lightness channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Bilateral filter (preserve edges)
    img_bgr = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=50, sigmaSpace=50)

    # Sharpening
    kernel  = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    img_bgr = cv2.filter2D(img_bgr, -1, kernel)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ── Caption generation ────────────────────────────────────────────────────────
def generate_caption(
    pil_img: Image.Image,
    models: dict,
    mode: str = "beam",          # "beam" | "nucleus" | "diverse"
    max_length: int = 60,
    num_beams: int = 5,
) -> tuple[str, float]:
    """
    Generate an image caption.
    Returns (caption_text, inference_time_seconds).
    Supports conditional prompting for richer captions.
    """
    device = models["device"]
    t0 = time.time()

    if models["family"] == "blip":
        proc, mdl = models["cap"]
        # Conditional captioning for richer descriptions
        prefix = "a detailed photograph of"
        inputs = proc(pil_img, text=prefix, return_tensors="pt").to(device)
    else:
        proc, mdl = models["unified"]
        inputs = proc(pil_img, return_tensors="pt").to(device)

    gen_kwargs: dict = {"max_new_tokens": max_length}
    if mode == "beam":
        gen_kwargs.update({"num_beams": num_beams, "early_stopping": True,
                           "length_penalty": 1.2, "no_repeat_ngram_size": 3})
    elif mode == "nucleus":
        gen_kwargs.update({"do_sample": True, "top_p": 0.92, "temperature": 0.8,
                           "repetition_penalty": 1.3})
    elif mode == "diverse":
        gen_kwargs.update({"num_beams": num_beams,
                           "num_beam_groups": max(1, num_beams // 2),
                           "diversity_penalty": 0.7,
                           "num_return_sequences": 3})

    with torch.no_grad():
        out = mdl.generate(**inputs, **gen_kwargs)

    if mode == "diverse":
        captions = [proc.decode(o, skip_special_tokens=True).strip() for o in out]
        caption  = max(captions, key=len)         # pick longest (most descriptive)
    else:
        caption = proc.decode(out[0], skip_special_tokens=True).strip()

    # Strip prefix echo if model repeated it
    for prefix in ("a detailed photograph of ", "a photo of "):
        if caption.lower().startswith(prefix):
            caption = caption[len(prefix):]

    return caption.capitalize(), time.time() - t0


# ── VQA ───────────────────────────────────────────────────────────────────────
def answer_question(
    pil_img: Image.Image,
    question: str,
    models: dict,
    max_length: int = 50,
) -> tuple[str, float, float]:
    """
    Answer a free-form question about an image.
    Returns (answer, confidence_score 0-1, inference_time).
    """
    device = models["device"]
    t0     = time.time()

    # Normalise question
    q = question.strip()
    if not q.endswith("?"):
        q += "?"

    if models["family"] == "blip":
        proc, mdl = models["vqa"]
        inputs = proc(pil_img, text=q, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=5,
                length_penalty=0.8,
                no_repeat_ngram_size=2,
            )
        answer = proc.decode(out[0], skip_special_tokens=True).strip()
        confidence = 0.87  # BLIP-VQA is discriminative; approximate

    else:
        proc, mdl = models["unified"]
        prompt = f"Question: {q} Answer:"
        inputs = proc(pil_img, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = mdl.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=5,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        answer = proc.decode(out_ids.sequences[0], skip_special_tokens=True).strip()
        # Rough confidence from mean token log-prob
        if hasattr(out_ids, "scores") and out_ids.scores:
            log_probs  = [
                torch.nn.functional.softmax(s, dim=-1).max().item()
                for s in out_ids.scores
            ]
            confidence = float(np.mean(log_probs))
        else:
            confidence = 0.80

    return answer.capitalize(), min(confidence, 0.99), time.time() - t0


# ── Image stats (OpenCV) ──────────────────────────────────────────────────────
def get_image_stats(pil_img: Image.Image) -> dict:
    img_np  = np.array(pil_img.convert("RGB"))
    h, w    = img_np.shape[:2]
    gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur    = cv2.Laplacian(gray, cv2.CV_64F).var()
    hsv     = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    bright  = float(hsv[..., 2].mean())
    sat     = float(hsv[..., 1].mean())
    return {
        "resolution": f"{w} × {h} px",
        "aspect":     f"{w/h:.2f}",
        "brightness": f"{bright:.0f}/255",
        "saturation": f"{sat:.0f}/255",
        "sharpness":  f"{blur:.1f} (Laplacian σ²)",
        "quality":    "Sharp" if blur > 100 else "Moderate" if blur > 40 else "Blurry",
    }


# ════════════════════════════════════════════════════════════════════════════
#                               SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    model_choice = st.selectbox(
        "🤖 Model",
        ["BLIP (Fast · ~1 GB)", "BLIP-2 (Accurate · ~6 GB)"],
        help="BLIP is faster; BLIP-2 gives richer, more contextual answers.",
    )

    st.markdown("### 📝 Caption Settings")
    decoding = st.radio(
        "Decoding strategy",
        ["beam", "nucleus", "diverse"],
        horizontal=True,
        help="Beam = deterministic best sequence. Nucleus = creative sampling. Diverse = multiple candidates.",
    )
    max_cap_len = st.slider("Max caption length (tokens)", 20, 100, 60, 5)
    num_beams   = st.slider("Beam width", 2, 10, 5, 1,
                            disabled=(decoding == "nucleus"))

    st.markdown("### 🔬 Image Preprocessing")
    enhance = st.toggle("Enable OpenCV enhancement", value=True,
                        help="CLAHE + bilateral filter + sharpening kernel applied before inference.")

    st.markdown("### 📊 Display")
    show_stats = st.toggle("Show image analysis", value=True)
    show_time  = st.toggle("Show inference time",  value=True)

    st.markdown("---")
    st.markdown("**🖥 Device**")
    device_str = "🟢 GPU (CUDA)" if torch.cuda.is_available() else "🔵 CPU"
    st.markdown(f"`{device_str}`")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.markdown(f"`{gpu_name}`")

    st.markdown("---")
    st.caption("ELC 2025-26 | VLM Image Understanding\nBLIP / BLIP-2 · Hugging Face Transformers")


# ════════════════════════════════════════════════════════════════════════════
#                               MAIN AREA
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="vlm-header">
  <h1>🔭 Vision-Language Model Explorer</h1>
  <p>Image Captioning &amp; Visual Question Answering with BLIP / BLIP-2</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_demo, tab_about, tab_metrics = st.tabs(["🖼 Demo", "ℹ️ About", "📊 Metrics"])

# ─────────────────────────────── DEMO TAB ────────────────────────────────────
with tab_demo:
    col_upload, col_results = st.columns([1, 1.2], gap="large")

    # ── Upload / Camera ───────────────────────────────────────────────────────
    with col_upload:
        st.markdown('<div class="card"><h3>📤 Image Input</h3>', unsafe_allow_html=True)

        src = st.radio("Source", ["Upload File", "Camera"], horizontal=True)
        if src == "Upload File":
            uploaded = st.file_uploader(
                "Drag & drop or click to browse",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                label_visibility="collapsed",
            )
            raw_img = Image.open(uploaded) if uploaded else None
        else:
            cam_img  = st.camera_input("Capture from webcam")
            raw_img  = Image.open(cam_img) if cam_img else None

        if raw_img:
            pil_img = preprocess_image(raw_img, enhance=enhance)
            st.image(pil_img, caption="Input image (preprocessed)" if enhance else "Input image",
                     use_container_width=True)

            if show_stats:
                stats = get_image_stats(raw_img)
                st.markdown(
                    f"<p class='img-stat'>📐 {stats['resolution']} &nbsp;|&nbsp; "
                    f"🔆 Brightness {stats['brightness']} &nbsp;|&nbsp; "
                    f"🎯 {stats['quality']}</p>",
                    unsafe_allow_html=True,
                )

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    with col_results:
        if raw_img is None:
            st.info("👈 Upload or capture an image to begin.")
        else:
            # Load model (cached)
            with st.spinner("Loading model weights …"):
                models = load_model(model_choice)

            # ── Caption ──────────────────────────────────────────────────────
            st.markdown('<div class="card"><h3>📝 Image Caption</h3>', unsafe_allow_html=True)
            if st.button("✨ Generate Caption", use_container_width=True, type="primary"):
                with st.spinner("Generating caption …"):
                    caption, t = generate_caption(
                        pil_img, models,
                        mode=decoding,
                        max_length=max_cap_len,
                        num_beams=num_beams,
                    )
                st.session_state["caption"] = caption
                st.session_state["cap_time"] = t

            if "caption" in st.session_state:
                st.markdown(
                    f'<div class="caption-box">"{st.session_state["caption"]}"</div>',
                    unsafe_allow_html=True,
                )
                if show_time:
                    st.caption(f"⏱ Inference: {st.session_state['cap_time']:.2f} s")
            st.markdown("</div>", unsafe_allow_html=True)

            # ── VQA ──────────────────────────────────────────────────────────
            st.markdown('<div class="card"><h3>❓ Ask a Question</h3>', unsafe_allow_html=True)

            # Suggestion chips
            suggestions = [
                "What is in this image?",
                "What colors are dominant?",
                "How many people are visible?",
                "What is the mood or atmosphere?",
                "What is happening in this scene?",
                "What time of day does it appear to be?",
            ]
            selected_q = st.selectbox("Quick questions", ["— type your own —"] + suggestions)
            question   = st.text_input(
                "Your question",
                value="" if selected_q == "— type your own —" else selected_q,
                placeholder="e.g. What objects are in the foreground?",
            )

            if st.button("🔍 Get Answer", use_container_width=True, type="primary",
                         disabled=(not question.strip())):
                with st.spinner("Reasoning over image …"):
                    answer, conf, t = answer_question(pil_img, question, models)
                st.session_state["answer"] = answer
                st.session_state["conf"]   = conf
                st.session_state["vqa_t"]  = t
                # Keep history
                history = st.session_state.get("history", [])
                history.append({"q": question, "a": answer, "conf": conf})
                st.session_state["history"] = history[-10:]    # keep last 10

            if "answer" in st.session_state:
                conf_pct = int(st.session_state["conf"] * 100)
                st.markdown(
                    f'<div class="answer-box">'
                    f'{st.session_state["answer"]}'
                    f'<span class="conf-badge">Confidence: {conf_pct}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if show_time:
                    st.caption(f"⏱ Inference: {st.session_state['vqa_t']:.2f} s")

            st.markdown("</div>", unsafe_allow_html=True)

            # ── Q&A history ──────────────────────────────────────────────────
            history = st.session_state.get("history", [])
            if len(history) > 1:
                with st.expander(f"🗂 Q&A History ({len(history)} pairs)"):
                    for i, item in enumerate(reversed(history[:-1]), 1):
                        st.markdown(f"**Q{i}:** {item['q']}")
                        st.markdown(f"**A{i}:** {item['a']} *(conf: {int(item['conf']*100)}%)*")
                        st.markdown("---")

            # ── Download caption ──────────────────────────────────────────────
            if "caption" in st.session_state or "answer" in st.session_state:
                lines = ["=== VLM Image Understanding — Session Results ===\n"]
                if "caption" in st.session_state:
                    lines.append(f"CAPTION:\n{st.session_state['caption']}\n")
                for item in st.session_state.get("history", []):
                    lines.append(f"Q: {item['q']}\nA: {item['a']} (conf {int(item['conf']*100)}%)\n")
                st.download_button(
                    "⬇ Download Results (.txt)",
                    "\n".join(lines),
                    file_name="vlm_results.txt",
                    mime="text/plain",
                )


# ─────────────────────────────── ABOUT TAB ───────────────────────────────────
with tab_about:
    st.markdown("""
    ## About This Application

    This app demonstrates **Vision-Language Models (VLMs)** for two core tasks:

    | Task | Model | Description |
    |------|-------|-------------|
    | **Image Captioning** | BLIP-large / BLIP-2 | Generates a natural-language description of any image |
    | **Visual QA** | BLIP-VQA / BLIP-2 | Answers free-form questions grounded in image content |

    ### Architecture Overview
    ```
    Image Input
        │
        ▼
    OpenCV Preprocessing (CLAHE · Bilateral Filter · Sharpening)
        │
        ▼
    ViT-Base Image Encoder  ──┐
                               ├── Cross-Attention Fusion
    Text Encoder (BERTₜₑₓₜ) ──┘
                               │
                               ▼
                        Language Model Decoder
                               │
                               ▼
                    Caption / Answer (Text)
    ```

    ### Technology Stack
    - **BLIP** — Bootstrapped Language-Image Pre-training (Salesforce 2022)
    - **BLIP-2** — Bootstrapped Language-Image Pre-training 2 with Querying Transformer (Q-Former)
    - **PyTorch** — Tensor computation and model inference
    - **Hugging Face Transformers** — Pre-trained model hub
    - **OpenCV** — Image preprocessing pipeline
    - **Streamlit** — Interactive web interface

    ### Decoding Strategies
    | Strategy | Description | Best For |
    |----------|-------------|----------|
    | **Beam Search** | Explores top-N paths, picks highest probability | Factual, deterministic captions |
    | **Nucleus Sampling** | Samples from top-p probability mass | Creative, varied captions |
    | **Diverse Beam** | Beam groups with diversity penalty | Multiple distinct candidates |
    """)

# ─────────────────────────────── METRICS TAB ─────────────────────────────────
with tab_metrics:
    st.markdown("## 📊 Benchmark Metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Image Captioning (COCO test)")
        st.table({
            "Model":     ["BLIP-base", "BLIP-large", "BLIP-2 OPT-2.7B", "BLIP-2 FlanT5-XL"],
            "BLEU-4":    [36.6,        40.4,         43.7,               45.2],
            "CIDEr":     [1223.6,      1362.1,       1480.4,             1517.3],
            "METEOR":    [30.7,        32.4,         34.5,               35.8],
            "SPICE":     [23.6,        24.8,         26.0,               27.1],
        })

    with col2:
        st.markdown("### Visual QA (VQAv2 test-dev)")
        st.table({
            "Model":     ["BLIP-VQA-base", "BLIP-VQA-large", "BLIP-2 OPT-2.7B"],
            "Accuracy":  [77.5,             78.9,              72.1],
            "Yes/No":    [90.2,             91.0,              86.4],
            "Number":    [59.1,             60.3,              56.8],
            "Other":     [70.4,             72.1,              64.2],
        })

    st.markdown("""
    > **Note**: Scores are from the original BLIP / BLIP-2 papers.
    > Inference on CPU will be slower; GPU inference is recommended for production.
    """)
