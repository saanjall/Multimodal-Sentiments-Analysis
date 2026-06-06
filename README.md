# 🔭 Vision-Language Models for Image Understanding

**ELC Project 2025-26 | Topic: Vision-Language Models (Image Captioning + VQA)**

A production-ready Streamlit application that demonstrates **Image Captioning** and **Visual Question Answering** using Salesforce BLIP and BLIP-2 models from Hugging Face Transformers.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📤 Image Upload | Drag-and-drop or camera capture |
| 📝 Image Captioning | Beam, nucleus, or diverse beam decoding |
| ❓ Visual QA | Free-form questions answered with confidence score |
| 🔬 Image Enhancement | CLAHE + bilateral filter + sharpening via OpenCV |
| 📊 Image Analysis | Resolution, brightness, sharpness stats |
| 🗂 Q&A History | Track up to 10 prior question-answer pairs |
| ⬇ Export | Download session results as .txt |

---

## 🏗 Project Structure

```
vlm_project/
├── src/
│   └── app.py              # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🚀 Quick Start

### 1. Clone / Set Up

```bash
git clone <your-repo>
cd vlm_project
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Install the CUDA-compatible PyTorch from https://pytorch.org/get-started/locally/

### 4. Run the App

```bash
streamlit run src/app.py
```

Open **http://localhost:8501** in your browser.

---

## 🤖 Models Used

### BLIP (Bootstrapped Language-Image Pre-training)
- **Caption**: `Salesforce/blip-image-captioning-large` (~990 MB)
- **VQA**: `Salesforce/blip-vqa-base` (~950 MB)
- Best for: Fast inference, CPU-compatible

### BLIP-2 (BLIP with Querying Transformer)
- **Unified**: `Salesforce/blip2-opt-2.7b` (~5.5 GB)
- Best for: Richer captions, conversational VQA, GPU required

Models are **automatically downloaded** on first run and cached by Hugging Face.

---

## 🔬 Image Preprocessing Pipeline (OpenCV)

```
Raw Image
    │
    ▼
BGR → LAB colour space
    │
    ▼
CLAHE (Contrast Limited Adaptive Histogram Equalization)
    │
    ▼
Bilateral Filter (edge-preserving denoising)
    │
    ▼
Sharpening Kernel
    │
    ▼
Preprocessed RGB Image → BLIP/BLIP-2
```

---

## 📊 Captioning Decoding Strategies

| Strategy | Method | Use Case |
|----------|--------|----------|
| **Beam Search** | `num_beams=5`, `length_penalty=1.2` | Factual, consistent captions |
| **Nucleus Sampling** | `top_p=0.92`, `temperature=0.8` | Creative, varied descriptions |
| **Diverse Beam** | Beam groups + diversity penalty | Multiple candidate captions |

---

## 🖥 Hardware Requirements

| Config | RAM | VRAM | Speed |
|--------|-----|------|-------|
| CPU (BLIP) | 4 GB | — | ~8-15s/image |
| GPU (BLIP) | 4 GB | 2 GB | ~0.5-1s/image |
| GPU (BLIP-2) | 8 GB | 6 GB | ~1-3s/image |

---

## 📚 References

1. Li, J. et al. "BLIP: Bootstrapping Language-Image Pre-training." ICML 2022.
2. Li, J. et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." ICML 2023.
3. Hugging Face Transformers: https://huggingface.co/docs/transformers
4. Streamlit: https://docs.streamlit.io

---

*ELC 2025-26 | Computer Vision & Deep Learning*
